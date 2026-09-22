//! Check GitHub releases and replace this binary.
//!
//! The running process is never restarted. The new file is swapped into the
//! install path and this session keeps the old image mapped, so the dashboard
//! stays usable. A build launched from `target/debug` or `target/release` is
//! left alone. Bytes are accepted only from GitHub over HTTPS, only after the
//! published sha256 matches, and only when the archive holds one executable
//! with the magic bytes for this operating system.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const ARCHIVE_LIMIT: u64 = 64 * 1024 * 1024;
const CHECKSUM_LIMIT: u64 = 8 * 1024;
const BINARY_LIMIT: u64 = 128 * 1024 * 1024;
/// Decompressed tar ceiling. The archive download itself is capped separately,
/// so a gzip bomb cannot expand without bound while we look for the binary.
const DECOMPRESSED_LIMIT: u64 = 256 * 1024 * 1024;
const MAX_ENTRIES: usize = 32;
const MAX_URL_LEN: usize = 16 * 1024;

const ALLOWED_HOSTS: &[&str] = &[
    "api.github.com",
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "github-releases.githubusercontent.com",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl Version {
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        let text = text
            .strip_prefix('v')
            .or_else(|| text.strip_prefix('V'))
            .unwrap_or(text);
        let mut parts = text.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some(Self {
            major,
            minor,
            patch,
        })
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub version: Version,
    pub archive_name: String,
    archive_url: String,
    checksum_url: String,
}

pub enum Outcome {
    /// Up to date, offline, or a source-tree build. Say nothing.
    Quiet,
    /// A newer release exists but replacing this copy is not safe.
    Notice(String),
    Offer(Plan),
}

/// Short enough for the status row when the banner does not fit.
pub fn clip_note(text: String) -> String {
    const MAX: usize = 180;
    if text.chars().count() <= MAX {
        text
    } else {
        let trimmed: String = text.chars().take(MAX.saturating_sub(1)).collect();
        format!("{trimmed}…")
    }
}

/// `cargo run` and `cargo test` place the executable under `target/`.
/// Replacing those files would fight the next build.
pub fn is_cargo_artifact(path: &Path) -> bool {
    let comps: Vec<&str> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    let Some(target_at) = comps.iter().rposition(|c| *c == "target") else {
        return false;
    };
    let after = &comps[target_at + 1..];
    if after.len() < 2 {
        return false;
    }
    let dirs = &after[..after.len() - 1];
    match dirs {
        ["release"] | ["debug"] => true,
        ["release", "deps"] | ["debug", "deps"] => true,
        [triple, "release"] | [triple, "debug"] if looks_like_triple(triple) => true,
        [triple, "release", "deps"] | [triple, "debug", "deps"] if looks_like_triple(triple) => {
            true
        }
        _ => false,
    }
}

fn looks_like_triple(value: &str) -> bool {
    let parts: Vec<&str> = value.split('-').collect();
    (3..=4).contains(&parts.len())
        && parts.iter().all(|part| {
            !part.is_empty() && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}

pub fn running_from_cargo() -> bool {
    std::env::current_exe()
        .ok()
        .map(|path| is_cargo_artifact(&path))
        .unwrap_or(false)
}

/// Drop a Windows leftover from the previous swap. The new process has
/// already started, so the file it replaced is no longer the running image.
pub fn cleanup_previous() {
    let Ok(exe) = locate_exe() else {
        return;
    };
    if is_cargo_artifact(&exe) {
        return;
    }
    if let Some(old) = previous_binary(&exe) {
        let _ = fs::remove_file(old);
    }
}

fn previous_binary(exe: &Path) -> Option<PathBuf> {
    let name = exe.file_name()?.to_str()?;
    Some(exe.with_file_name(format!("{name}.old")))
}

pub fn check() -> Outcome {
    let exe = match locate_exe() {
        Ok(path) => path,
        Err(_) => return Outcome::Quiet,
    };
    if is_cargo_artifact(&exe) {
        return Outcome::Quiet;
    }
    let Some(current) = Version::parse(env!("CARGO_PKG_VERSION")) else {
        return Outcome::Quiet;
    };
    let Some((owner, repo)) = github_repo(env!("CARGO_PKG_REPOSITORY")) else {
        return Outcome::Quiet;
    };
    let url = format!("https://api.github.com/repos/{owner}/{repo}/releases/latest");
    let body = match http_get(&url, 2 * 1024 * 1024, true) {
        Ok(body) => body,
        Err(_) => return Outcome::Quiet,
    };
    let text = match String::from_utf8(body) {
        Ok(text) => text,
        Err(_) => return Outcome::Quiet,
    };
    let parsed = match parse_release(&text) {
        Ok(parsed) => parsed,
        Err(_) => return Outcome::Quiet,
    };
    if parsed.version <= current {
        return Outcome::Quiet;
    }
    let version = parsed.version;
    let Some(target) = release_target() else {
        return Outcome::Notice(format!(
            "release {version} is available — no build is published for this platform"
        ));
    };
    let plan = match select_plan(version, &parsed.assets, target) {
        Ok(plan) => plan,
        Err(err) => {
            return Outcome::Notice(format!("release {version} is available — {err}"));
        }
    };
    if let Err(err) = ensure_writable(&exe) {
        return Outcome::Notice(format!("release {version} is available — {err}"));
    }
    Outcome::Offer(plan)
}

pub fn install(plan: &Plan) -> Result<String, String> {
    let exe = locate_exe()?;
    if is_cargo_artifact(&exe) {
        return Err("this copy was built from source and was not replaced".into());
    }
    ensure_writable(&exe)?;
    url_allowed(&plan.archive_url)?;
    url_allowed(&plan.checksum_url)?;
    let checksum = http_get(&plan.checksum_url, CHECKSUM_LIMIT, false)?;
    let archive = http_get(&plan.archive_url, ARCHIVE_LIMIT, false)?;
    let checksum =
        String::from_utf8(checksum).map_err(|_| "checksum file is not text".to_string())?;
    let checksum = checksum.trim_start_matches('\u{feff}');
    verify_archive(&archive, checksum, &plan.archive_name)?;
    let binary_name = binary_name_for(std::env::consts::OS);
    let binary = extract_binary(&archive, &plan.archive_name, binary_name)?;
    if !magic_ok(&binary, std::env::consts::OS) {
        return Err("downloaded file is not an executable for this system".into());
    }
    replace_file(&exe, &binary)?;
    Ok(format!("updated to {} — relaunch to use it", plan.version))
}

pub fn dismissed(version: &Version) -> bool {
    let Some(path) = dismiss_path() else {
        return false;
    };
    match fs::read_to_string(path) {
        Ok(text) => text_dismisses(&text, version),
        Err(_) => false,
    }
}

pub fn dismiss(version: &Version) -> Result<(), String> {
    let path = dismiss_path().ok_or("no home directory to remember the dismissal")?;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|err| format!("{}: {err}", dir.display()))?;
    }
    fs::write(&path, format!("{version}\n")).map_err(|err| format!("{}: {err}", path.display()))
}

fn dismiss_path() -> Option<PathBuf> {
    crate::settings::data_dir().map(|dir| dir.join("skipped-release"))
}

fn text_dismisses(text: &str, version: &Version) -> bool {
    text.lines()
        .next()
        .map(|line| line.trim() == version.to_string())
        .unwrap_or(false)
}

fn github_repo(repository_url: &str) -> Option<(String, String)> {
    let trimmed = repository_url
        .trim()
        .trim_end_matches('/')
        .trim_end_matches(".git");
    let rest = trimmed
        .strip_prefix("https://github.com/")
        .or_else(|| trimmed.strip_prefix("http://github.com/"))
        .or_else(|| trimmed.strip_prefix("git@github.com:"))?;
    let (owner, name) = rest.split_once('/')?;
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        return None;
    }
    let ident = |value: &str| {
        !value.is_empty()
            && value
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    };
    if !ident(owner) || !ident(name) {
        return None;
    }
    Some((owner.to_string(), name.to_string()))
}

fn release_target() -> Option<&'static str> {
    release_target_for(std::env::consts::OS, std::env::consts::ARCH)
}

fn release_target_for(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("linux", "x86_64") => Some("linux-x86_64"),
        ("linux", "aarch64") => Some("linux-arm64"),
        ("macos", "x86_64") => Some("macos-x86_64"),
        ("macos", "aarch64") => Some("macos-arm64"),
        ("windows", "x86_64") => Some("windows-x86_64"),
        ("windows", "aarch64") => Some("windows-arm64"),
        _ => None,
    }
}

fn binary_name_for(os: &str) -> &'static str {
    if os == "windows" {
        "llm-visuals.exe"
    } else {
        "llm-visuals"
    }
}

fn archive_name(version: &Version, target: &str) -> String {
    let ext = if target.starts_with("windows-") {
        "zip"
    } else {
        "tar.gz"
    };
    format!("llm-visuals-{version}-{target}.{ext}")
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    assets: Vec<GhAsset>,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

struct ParsedRelease {
    version: Version,
    assets: Vec<GhAsset>,
}

fn parse_release(body: &str) -> Result<ParsedRelease, String> {
    let release: GhRelease =
        serde_json::from_str(body).map_err(|_| "unrecognised release response".to_string())?;
    if release.draft || release.prerelease {
        return Err("not a stable release".into());
    }
    let version = Version::parse(&release.tag_name).ok_or("release tag is not a version")?;
    Ok(ParsedRelease {
        version,
        assets: release.assets,
    })
}

fn select_plan(version: Version, assets: &[GhAsset], target: &str) -> Result<Plan, String> {
    let archive_name = archive_name(&version, target);
    let checksum_name = format!("{archive_name}.sha256");
    let archive = find_asset(assets, &archive_name)?;
    let checksum = find_asset(assets, &checksum_name)?;
    if archive.size > ARCHIVE_LIMIT || checksum.size > CHECKSUM_LIMIT {
        return Err("published file is larger than expected".into());
    }
    url_allowed(&archive.browser_download_url)?;
    url_allowed(&checksum.browser_download_url)?;
    Ok(Plan {
        version,
        archive_name,
        archive_url: archive.browser_download_url.clone(),
        checksum_url: checksum.browser_download_url.clone(),
    })
}

fn find_asset<'a>(assets: &'a [GhAsset], name: &str) -> Result<&'a GhAsset, String> {
    assets
        .iter()
        .find(|asset| asset.name == name)
        .ok_or_else(|| format!("no published file {name}"))
}

fn verify_archive(bytes: &[u8], checksum_text: &str, archive_name: &str) -> Result<(), String> {
    let expected = parse_checksum(checksum_text, archive_name)?;
    let got = Sha256::digest(bytes);
    if got.as_slice() != expected.as_slice() {
        return Err("checksum does not match the downloaded archive".into());
    }
    Ok(())
}

fn parse_checksum(text: &str, expected_name: &str) -> Result<[u8; 32], String> {
    let mut found = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let hex_hash = parts
            .next()
            .ok_or_else(|| "checksum line has no hash".to_string())?;
        let name = parts
            .next()
            .ok_or_else(|| "checksum line has no filename".to_string())?;
        if parts.next().is_some() {
            return Err("checksum line has extra fields".into());
        }
        let name = name.trim_start_matches('*');
        let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
        if base == expected_name {
            if found.is_some() {
                return Err("checksum lists the archive twice".into());
            }
            found = Some(parse_hex32(hex_hash)?);
        }
    }
    found.ok_or_else(|| format!("checksum file does not list {expected_name}"))
}

fn parse_hex32(text: &str) -> Result<[u8; 32], String> {
    if text.len() != 64 {
        return Err("checksum is not a sha256 hex string".into());
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&text[start..start + 2], 16)
            .map_err(|_| "checksum is not a sha256 hex string".to_string())?;
    }
    Ok(out)
}

enum EntryKind {
    Binary,
    Other,
}

/// `Ok(Binary)` only for the single expected file at the archive root.
/// Any `..`, absolute path, or drive-qualified name rejects the whole archive.
fn classify_entry(name: &str, binary_name: &str) -> Result<EntryKind, String> {
    let name = name.replace('\\', "/");
    if name.starts_with('/') || name.contains(':') || name.contains('\0') {
        return Err(format!("archive path is not safe: {name}"));
    }
    let mut parts = Vec::new();
    for part in name.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err(format!("archive path is not safe: {name}"));
        }
        parts.push(part);
    }
    if parts.len() == 1 && parts[0] == binary_name {
        Ok(EntryKind::Binary)
    } else {
        Ok(EntryKind::Other)
    }
}

fn extract_binary(
    archive: &[u8],
    archive_name: &str,
    binary_name: &str,
) -> Result<Vec<u8>, String> {
    if archive_name.ends_with(".tar.gz") {
        extract_tar_gz(archive, binary_name)
    } else if archive_name.ends_with(".zip") {
        extract_zip(archive, binary_name)
    } else {
        Err("unknown archive type".into())
    }
}

fn extract_tar_gz(bytes: &[u8], binary_name: &str) -> Result<Vec<u8>, String> {
    let decoder = flate2::read::GzDecoder::new(io::Cursor::new(bytes));
    let decoder = decoder.take(DECOMPRESSED_LIMIT);
    let mut archive = tar::Archive::new(decoder);
    let mut found = None;
    let mut count = 0usize;
    let entries = archive
        .entries()
        .map_err(|_| "archive could not be read".to_string())?;
    for entry in entries {
        count += 1;
        if count > MAX_ENTRIES {
            return Err("archive has too many entries".into());
        }
        let mut entry = entry.map_err(|_| "archive could not be read".to_string())?;
        let path = entry
            .path()
            .map_err(|_| "archive could not be read".to_string())?;
        let name = path.to_string_lossy();
        let kind = classify_entry(&name, binary_name)?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        if matches!(kind, EntryKind::Binary) {
            if found.is_some() {
                return Err("archive contains more than one executable".into());
            }
            found = Some(read_capped(&mut entry, BINARY_LIMIT)?);
        }
    }
    found.ok_or_else(|| format!("archive has no {binary_name}"))
}

fn extract_zip(bytes: &[u8], binary_name: &str) -> Result<Vec<u8>, String> {
    let mut archive = zip::ZipArchive::new(io::Cursor::new(bytes))
        .map_err(|_| "archive could not be read".to_string())?;
    if archive.len() > MAX_ENTRIES {
        return Err("archive has too many entries".into());
    }
    let mut found = None;
    for index in 0..archive.len() {
        let mut file = archive
            .by_index(index)
            .map_err(|_| "archive could not be read".to_string())?;
        let name = file.name().to_string();
        let kind = classify_entry(&name, binary_name)?;
        if file.is_dir() {
            continue;
        }
        if matches!(kind, EntryKind::Binary) {
            if found.is_some() {
                return Err("archive contains more than one executable".into());
            }
            found = Some(read_capped(&mut file, BINARY_LIMIT)?);
        }
    }
    found.ok_or_else(|| format!("archive has no {binary_name}"))
}

fn read_capped<R: Read>(reader: &mut R, limit: u64) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    reader
        .take(limit + 1)
        .read_to_end(&mut buf)
        .map_err(|_| "reading archive entry failed".to_string())?;
    if buf.len() as u64 > limit {
        return Err("executable inside the archive is larger than expected".into());
    }
    if buf.is_empty() {
        return Err("executable inside the archive is empty".into());
    }
    Ok(buf)
}

fn magic_ok(bytes: &[u8], os: &str) -> bool {
    if bytes.len() < 64 {
        return false;
    }
    match os {
        "linux" => bytes.starts_with(b"\x7fELF"),
        "windows" => bytes.starts_with(b"MZ"),
        "macos" => matches!(
            &bytes[..4],
            [0xFE, 0xED, 0xFA, 0xCE]
                | [0xFE, 0xED, 0xFA, 0xCF]
                | [0xCE, 0xFA, 0xED, 0xFE]
                | [0xCF, 0xFA, 0xED, 0xFE]
                | [0xCA, 0xFE, 0xBA, 0xBE]
                | [0xCA, 0xFE, 0xBA, 0xBF]
                | [0xBE, 0xBA, 0xFE, 0xCA]
        ),
        _ => false,
    }
}

fn locate_exe() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|_| "cannot find the running binary".to_string())?;
    // Follow links so a symlink in `~/bin` updates the real file, and so a
    // link into `target/release` is recognised as a source build.
    match fs::canonicalize(&exe) {
        Ok(path) => Ok(path),
        Err(_) => Ok(exe),
    }
}

fn ensure_writable(exe: &Path) -> Result<(), String> {
    let dir = exe
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .ok_or_else(|| format!("cannot replace {} (no directory)", exe.display()))?;
    let probe = dir.join(format!(".llm-visuals-write-probe-{}", std::process::id()));
    let _ = fs::remove_file(&probe);
    match OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(file) => {
            drop(file);
            let _ = fs::remove_file(&probe);
            Ok(())
        }
        Err(err) => Err(format!("cannot replace {} ({err})", exe.display())),
    }
}

/// Write `bytes` over `dest`. On Unix the rename onto the existing path is
/// atomic and the previous inode stays mapped in this process. On Windows the
/// running file has to be moved aside first; a failed second rename puts it back.
fn replace_file(dest: &Path, bytes: &[u8]) -> Result<(), String> {
    let dir = dest
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .ok_or_else(|| "binary has no directory".to_string())?;
    let name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "binary name is not utf-8".to_string())?;
    let tmp = dir.join(format!(".{name}.{}.new", std::process::id()));
    if let Err(err) = write_new(&tmp, bytes) {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    if let Err(err) = copy_permissions(dest, &tmp) {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    if let Err(err) = swap_in(dest, &tmp) {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("could not write the new binary ({err})"))?;
    file.write_all(bytes)
        .map_err(|err| format!("could not write the new binary ({err})"))?;
    file.sync_all()
        .map_err(|err| format!("could not write the new binary ({err})"))?;
    Ok(())
}

fn copy_permissions(from: &Path, to: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut mode = fs::metadata(from)
            .map(|meta| meta.permissions().mode() & 0o777)
            .unwrap_or(0o755);
        if mode & 0o111 == 0 {
            mode |= 0o755;
        }
        let mut perms = fs::metadata(to)
            .map_err(|err| format!("could not set permissions ({err})"))?
            .permissions();
        perms.set_mode(mode);
        fs::set_permissions(to, perms)
            .map_err(|err| format!("could not set permissions ({err})"))?;
    }
    #[cfg(not(unix))]
    {
        let _ = (from, to);
    }
    Ok(())
}

fn swap_in(dest: &Path, tmp: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        fs::rename(tmp, dest).map_err(|err| format!("could not replace the binary ({err})"))
    }
    #[cfg(windows)]
    {
        let Some(old) = previous_binary(dest) else {
            return Err("binary name is not utf-8".into());
        };
        let _ = fs::remove_file(&old);
        if let Err(err) = fs::rename(dest, &old) {
            return Err(format!("could not move the current binary aside ({err})"));
        }
        if let Err(err) = fs::rename(tmp, dest) {
            let rolled = fs::rename(&old, dest);
            return Err(match rolled {
                Ok(()) => format!("could not install the new binary, original restored ({err})"),
                Err(restore) => format!(
                    "could not install the new binary ({err}) and could not restore the original ({restore})"
                ),
            });
        }
        let _ = fs::remove_file(&old);
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (dest, tmp);
        Err("updating in place is not supported on this platform".into())
    }
}

fn url_allowed(raw: &str) -> Result<url::Url, String> {
    if raw.len() > MAX_URL_LEN {
        return Err("download url is unreasonably long".into());
    }
    let url = url::Url::parse(raw).map_err(|_| "download url is invalid".to_string())?;
    if url.scheme() != "https" {
        return Err("refusing a download that is not https".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("refusing a download url that carries credentials".into());
    }
    if url.port().is_some() {
        return Err("refusing a download on a non-default port".into());
    }
    let host = url.host_str().unwrap_or("");
    if !ALLOWED_HOSTS.contains(&host) {
        return Err(format!("refusing download host {host}"));
    }
    Ok(url)
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .redirects(0)
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(120))
        .timeout(Duration::from_secs(180))
        .build()
}

fn user_agent() -> String {
    format!("llm-visuals/{}", env!("CARGO_PKG_VERSION"))
}

/// Redirects are followed by hand so every hop is checked against the host
/// list. ureq's error text includes the request URL, and release redirects
/// carry a signed query, so those strings never reach the status line.
fn http_get(start: &str, limit: u64, api: bool) -> Result<Vec<u8>, String> {
    let agent = agent();
    let mut current = start.to_string();
    for _ in 0..5 {
        let url = url_allowed(&current)?;
        let request = agent.get(url.as_str()).set("User-Agent", &user_agent());
        let request = if api {
            request
                .set("Accept", "application/vnd.github+json")
                .set("X-GitHub-Api-Version", "2022-11-28")
        } else {
            request
        };
        match request.call() {
            Ok(response) => return read_body(response, limit),
            Err(ureq::Error::Status(code, response))
                if matches!(code, 301 | 302 | 303 | 307 | 308) =>
            {
                let location = response
                    .header("location")
                    .ok_or_else(|| "download redirect had no location".to_string())?
                    .to_string();
                current = join_url(url.as_str(), &location)?;
            }
            Err(ureq::Error::Status(code, _)) => {
                return Err(format!("download failed: HTTP {code}"));
            }
            Err(ureq::Error::Transport(err)) => {
                let msg = err.to_string().to_ascii_lowercase();
                if msg.contains("timed") {
                    return Err("download timed out".into());
                }
                return Err("download failed".into());
            }
        }
    }
    Err("download was redirected too many times".into())
}

fn join_url(base: &str, location: &str) -> Result<String, String> {
    if location.len() > MAX_URL_LEN {
        return Err("download redirect is unreasonably long".into());
    }
    let base = url::Url::parse(base).map_err(|_| "download url is invalid".to_string())?;
    let next = base
        .join(location)
        .map_err(|_| "download redirect is invalid".to_string())?;
    Ok(next.to_string())
}

fn read_body(response: ureq::Response, limit: u64) -> Result<Vec<u8>, String> {
    if let Some(len) = response
        .header("content-length")
        .and_then(|value| value.parse::<u64>().ok())
    {
        if len > limit {
            return Err("download is larger than expected".into());
        }
    }
    let mut buf = Vec::new();
    response
        .into_reader()
        .take(limit + 1)
        .read_to_end(&mut buf)
        .map_err(|_| "download failed".to_string())?;
    if buf.len() as u64 > limit {
        return Err("download is larger than expected".into());
    }
    if buf.is_empty() {
        return Err("download was empty".into());
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_numerically() {
        let parse = |text| Version::parse(text).unwrap();
        assert!(parse("v0.8.3") > parse("0.8.2"));
        assert!(parse("0.8.10") > parse("0.8.9"));
        assert!(parse("0.10.0") > parse("0.9.9"));
        assert_eq!(parse("V0.8.2"), parse("0.8.2"));
        assert!(Version::parse("0.8").is_none());
        assert!(Version::parse("0.8.2-rc1").is_none());
        assert!(Version::parse("1.2.3.4").is_none());
    }

    #[test]
    fn release_targets_match_the_published_names() {
        assert_eq!(release_target_for("linux", "x86_64"), Some("linux-x86_64"));
        assert_eq!(release_target_for("linux", "aarch64"), Some("linux-arm64"));
        assert_eq!(release_target_for("macos", "x86_64"), Some("macos-x86_64"));
        assert_eq!(release_target_for("macos", "aarch64"), Some("macos-arm64"));
        assert_eq!(
            release_target_for("windows", "x86_64"),
            Some("windows-x86_64")
        );
        assert_eq!(
            release_target_for("windows", "aarch64"),
            Some("windows-arm64")
        );
        assert_eq!(release_target_for("freebsd", "x86_64"), None);
        let version = Version::parse("0.8.2").unwrap();
        assert_eq!(
            archive_name(&version, "linux-x86_64"),
            "llm-visuals-0.8.2-linux-x86_64.tar.gz"
        );
        assert_eq!(
            archive_name(&version, "windows-arm64"),
            "llm-visuals-0.8.2-windows-arm64.zip"
        );
    }

    #[test]
    fn cargo_artifacts_are_recognised_and_installs_are_not() {
        assert!(is_cargo_artifact(Path::new(
            "/home/u/proj/target/release/llm-visuals"
        )));
        assert!(is_cargo_artifact(Path::new(
            "/home/u/proj/target/debug/llm-visuals"
        )));
        assert!(is_cargo_artifact(Path::new(
            "/home/u/proj/target/x86_64-unknown-linux-gnu/release/llm-visuals"
        )));
        assert!(is_cargo_artifact(Path::new(
            "/home/u/proj/target/debug/deps/llm_visuals-abc"
        )));
        assert!(!is_cargo_artifact(Path::new(
            "/home/u/.cargo/bin/llm-visuals"
        )));
        assert!(!is_cargo_artifact(Path::new("/usr/local/bin/llm-visuals")));
        let exe = std::env::current_exe().unwrap();
        assert!(
            is_cargo_artifact(&exe),
            "test binary should sit under target/: {}",
            exe.display()
        );
    }

    #[test]
    fn repository_url_parses_only_a_github_slug() {
        assert_eq!(
            github_repo("https://github.com/DingoOz/llm-visuals"),
            Some(("DingoOz".into(), "llm-visuals".into()))
        );
        assert_eq!(
            github_repo("git@github.com:DingoOz/llm-visuals.git"),
            Some(("DingoOz".into(), "llm-visuals".into()))
        );
        assert!(github_repo("https://example.com/DingoOz/llm-visuals").is_none());
        assert!(github_repo("https://github.com/DingoOz/llm-visuals/extra").is_none());
    }

    #[test]
    fn download_urls_stay_on_github_https() {
        let ok = "https://github.com/DingoOz/llm-visuals/releases/download/v0.8.2/llm-visuals-0.8.2-linux-x86_64.tar.gz";
        assert!(url_allowed(ok).is_ok());
        assert!(url_allowed("https://release-assets.githubusercontent.com/asset?sig=abc").is_ok());
        assert!(
            url_allowed("https://api.github.com/repos/DingoOz/llm-visuals/releases/latest").is_ok()
        );
        assert!(url_allowed("http://github.com/DingoOz/llm-visuals").is_err());
        assert!(url_allowed("https://user:pass@github.com/DingoOz/llm-visuals").is_err());
        assert!(url_allowed("https://github.com:8443/DingoOz/llm-visuals").is_err());
        assert!(
            url_allowed("https://release-assets.githubusercontent.com.evil.example/x").is_err()
        );
        assert!(url_allowed("https://evil.example/llm-visuals").is_err());
    }

    fn sample_assets(archive_url: &str, checksum_url: &str, size: u64) -> String {
        format!(
            r#"{{
                "tag_name": "v9.9.9",
                "draft": false,
                "prerelease": false,
                "assets": [
                    {{"name": "llm-visuals-9.9.9-linux-x86_64.tar.gz", "browser_download_url": "{archive_url}", "size": {size}}},
                    {{"name": "llm-visuals-9.9.9-linux-x86_64.tar.gz.sha256", "browser_download_url": "{checksum_url}", "size": 80}},
                    {{"name": "llm-visuals-9.9.9-windows-x86_64.zip", "browser_download_url": "https://github.com/DingoOz/llm-visuals/releases/download/v9.9.9/llm-visuals-9.9.9-windows-x86_64.zip", "size": 10}}
                ]
            }}"#
        )
    }

    #[test]
    fn release_json_selects_one_asset_and_rejects_the_rest() {
        let body = sample_assets(
            "https://github.com/DingoOz/llm-visuals/releases/download/v9.9.9/llm-visuals-9.9.9-linux-x86_64.tar.gz",
            "https://github.com/DingoOz/llm-visuals/releases/download/v9.9.9/llm-visuals-9.9.9-linux-x86_64.tar.gz.sha256",
            1000,
        );
        let parsed = parse_release(&body).unwrap();
        let plan = select_plan(parsed.version, &parsed.assets, "linux-x86_64").unwrap();
        assert_eq!(plan.version.to_string(), "9.9.9");
        assert_eq!(plan.archive_name, "llm-visuals-9.9.9-linux-x86_64.tar.gz");
        assert!(select_plan(parsed.version, &parsed.assets, "linux-arm64").is_err());

        let huge = sample_assets(
            "https://github.com/DingoOz/llm-visuals/releases/download/v9.9.9/llm-visuals-9.9.9-linux-x86_64.tar.gz",
            "https://github.com/DingoOz/llm-visuals/releases/download/v9.9.9/llm-visuals-9.9.9-linux-x86_64.tar.gz.sha256",
            ARCHIVE_LIMIT + 1,
        );
        let parsed = parse_release(&huge).unwrap();
        assert!(select_plan(parsed.version, &parsed.assets, "linux-x86_64").is_err());

        let evil = sample_assets(
            "https://evil.example/llm-visuals.tar.gz",
            "https://github.com/DingoOz/llm-visuals/releases/download/v9.9.9/llm-visuals-9.9.9-linux-x86_64.tar.gz.sha256",
            1000,
        );
        let parsed = parse_release(&evil).unwrap();
        assert!(select_plan(parsed.version, &parsed.assets, "linux-x86_64").is_err());

        let draft = r#"{"tag_name":"v9.9.9","draft":true,"prerelease":false,"assets":[]}"#;
        assert!(parse_release(draft).is_err());
    }

    #[test]
    fn checksum_must_name_the_archive_and_match_its_bytes() {
        let name = "llm-visuals-9.9.9-linux-x86_64.tar.gz";
        let bytes = b"archive-bytes";
        let digest = hex32(&Sha256::digest(bytes));
        let text = format!("{digest}  {name}\n");
        verify_archive(bytes, &text, name).unwrap();
        assert!(verify_archive(b"tampered", &text, name).is_err());
        let other = format!("{digest}  other.tar.gz\n");
        assert!(parse_checksum(&other, name).is_err());
        let binary_mode = format!("{digest} *{name}\n");
        assert!(parse_checksum(&binary_mode, name).is_ok());
        assert!(parse_checksum("abcd  file\n", name).is_err());
    }

    fn hex32(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn fake_elf() -> Vec<u8> {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes
    }

    fn gzip(raw: &[u8]) -> Vec<u8> {
        let mut gz = Vec::new();
        let mut encoder = flate2::write::GzEncoder::new(&mut gz, flate2::Compression::fast());
        encoder.write_all(raw).unwrap();
        encoder.finish().unwrap();
        gz
    }

    /// One-file ustar. Used for names the `tar` crate refuses to write.
    fn ustar(name: &str, bytes: &[u8]) -> Vec<u8> {
        assert!(name.len() < 100);
        let mut header = [0u8; 512];
        header[..name.len()].copy_from_slice(name.as_bytes());
        header[100..108].copy_from_slice(b"0000755\0");
        header[108..116].copy_from_slice(b"0000000\0");
        header[116..124].copy_from_slice(b"0000000\0");
        let size = format!("{:011o}\0", bytes.len());
        header[124..136].copy_from_slice(size.as_bytes());
        header[136..148].copy_from_slice(b"00000000000\0");
        header[148..156].copy_from_slice(b"        ");
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let sum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
        let checksum = format!("{sum:06o}\0 ");
        header[148..156].copy_from_slice(checksum.as_bytes());
        let mut out = header.to_vec();
        out.extend_from_slice(bytes);
        let pad = (512 - (bytes.len() % 512)) % 512;
        out.extend(std::iter::repeat(0).take(pad));
        out.extend(std::iter::repeat(0).take(1024));
        out
    }

    fn tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut raw = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut raw);
            for (name, bytes) in files {
                let mut header = tar::Header::new_gnu();
                header.set_mode(0o755);
                header.set_size(bytes.len() as u64);
                header.set_cksum();
                builder.append_data(&mut header, name, *bytes).unwrap();
            }
            builder.finish().unwrap();
        }
        gzip(&raw)
    }

    fn zip_files(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut cursor = io::Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            for (name, bytes) in files {
                writer.start_file(name, options).unwrap();
                writer.write_all(bytes).unwrap();
            }
            writer.finish().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn archives_yield_only_the_root_executable() {
        let bin = fake_elf();
        let tar = tar_gz(&[
            ("llm-visuals", &bin),
            ("README.txt", b"notes"),
            ("./llm-visuals-extra", b"nope"),
        ]);
        let got =
            extract_binary(&tar, "llm-visuals-9.9.9-linux-x86_64.tar.gz", "llm-visuals").unwrap();
        assert_eq!(got, bin);

        let zip = zip_files(&[("llm-visuals.exe", &bin)]);
        let got = extract_binary(
            &zip,
            "llm-visuals-9.9.9-windows-x86_64.zip",
            "llm-visuals.exe",
        )
        .unwrap();
        assert_eq!(got, bin);

        // The tar crate will not write these names, so the hostile headers are
        // built by hand. A well-formed hand-built archive must still extract,
        // which is what makes the refusals below about the path.
        let handmade = gzip(&ustar("llm-visuals", &bin));
        assert_eq!(
            extract_binary(
                &handmade,
                "llm-visuals-9.9.9-linux-x86_64.tar.gz",
                "llm-visuals"
            )
            .unwrap(),
            bin
        );
        let escaped = gzip(&ustar("../llm-visuals", &bin));
        assert!(extract_binary(
            &escaped,
            "llm-visuals-9.9.9-linux-x86_64.tar.gz",
            "llm-visuals"
        )
        .is_err());
        let absolute = gzip(&ustar("/tmp/llm-visuals", &bin));
        assert!(extract_binary(
            &absolute,
            "llm-visuals-9.9.9-linux-x86_64.tar.gz",
            "llm-visuals"
        )
        .is_err());
        let drive = gzip(&ustar("C:llm-visuals", &bin));
        assert!(extract_binary(
            &drive,
            "llm-visuals-9.9.9-linux-x86_64.tar.gz",
            "llm-visuals"
        )
        .is_err());
        let slipped = zip_files(&[("../llm-visuals.exe", &bin)]);
        assert!(extract_binary(
            &slipped,
            "llm-visuals-9.9.9-windows-x86_64.zip",
            "llm-visuals.exe"
        )
        .is_err());
        let doubled = tar_gz(&[("llm-visuals", &bin), ("llm-visuals", &bin)]);
        assert!(extract_binary(
            &doubled,
            "llm-visuals-9.9.9-linux-x86_64.tar.gz",
            "llm-visuals"
        )
        .is_err());
    }

    #[test]
    fn magic_matches_the_operating_system() {
        assert!(magic_ok(&fake_elf(), "linux"));
        let mut mz = vec![0u8; 64];
        mz[..2].copy_from_slice(b"MZ");
        assert!(magic_ok(&mz, "windows"));
        assert!(!magic_ok(&mz, "linux"));
        assert!(!magic_ok(&fake_elf()[..32], "linux"));
        let mut macho = vec![0u8; 64];
        macho[..4].copy_from_slice(&[0xCF, 0xFA, 0xED, 0xFE]);
        assert!(magic_ok(&macho, "macos"));
    }

    #[test]
    fn replace_file_swaps_contents_and_leaves_no_temp() {
        let dir =
            std::env::temp_dir().join(format!("llm-visuals-replace-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join(if cfg!(windows) {
            "llm-visuals.exe"
        } else {
            "llm-visuals"
        });
        fs::write(&dest, b"old-binary-contents").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&dest).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&dest, perms).unwrap();
        }
        replace_file(&dest, b"new-binary-contents").unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"new-binary-contents");
        let names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 1, "{names:?}");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&dest).unwrap().permissions().mode() & 0o777,
                0o755
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dismissal_text_is_one_version() {
        let version = Version::parse("0.8.3").unwrap();
        assert!(text_dismisses("0.8.3\n", &version));
        assert!(!text_dismisses("0.8.2\n", &version));
        assert!(!text_dismisses("", &version));
    }

    #[cfg(unix)]
    #[test]
    fn unwritable_directory_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir =
            std::env::temp_dir().join(format!("llm-visuals-nowrite-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("llm-visuals");
        fs::write(&dest, b"x").unwrap();
        let mut perms = fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o555);
        fs::set_permissions(&dir, perms).unwrap();
        let result = ensure_writable(&dest);
        let mut perms = fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&dir, perms).unwrap();
        let _ = fs::remove_dir_all(&dir);
        if result.is_ok() {
            // Root ignores the mode bits, so the probe still succeeds.
            return;
        }
        assert!(result.is_err());
    }
}
