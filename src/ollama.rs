//! Ollama: the daemon's `GET /api/ps` and the model store on disk.
//!
//! Current Ollama runs each loaded GGML model in a bundled `llama-server`
//! (`…/lib/ollama/llama-server --model …/blobs/sha256-…`), which the
//! llama.cpp poller already reads through `/slots`. What only the daemon
//! knows comes from `/api/ps`: the model's tag, how much of it sits in VRAM,
//! its context length and when it will be unloaded. Older releases ran
//! `ollama runner`, whose port has no token counters; there `/api/ps` is all
//! the dashboard can show, and the throughput panel says so.

use std::path::{Path, PathBuf};

use crate::model_detect::DetectedModel;
use crate::observe::{http_get, HttpAuth};

pub const DEFAULT_PORT: u16 = 11434;

/// The layer of a manifest that holds the weights.
const MODEL_LAYER: &str = "application/vnd.ollama.image.model";
const DEFAULT_REGISTRY: &str = "registry.ollama.ai";
const DEFAULT_NAMESPACE: &str = "library";

/// A keep-alive this far out is `keep_alive: -1`: Ollama turns a negative
/// duration into the largest one it can hold (about 292 years).
const FOREVER_SECS: i64 = 50 * 365 * 24 * 3600;

/// One entry of `/api/ps`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PsModel {
    pub name: String,
    /// Bytes resident in total: weights, KV cache and compute buffers.
    pub size: u64,
    /// The part of `size` held on GPUs.
    pub size_vram: u64,
    pub context_length: usize,
    /// Unix seconds; `None` when absent or Go's zero time.
    pub expires_at: Option<i64>,
    pub family: String,
    pub parameter_size: String,
    pub quantization: String,
}

impl PsModel {
    /// Bytes in system RAM rather than on a GPU. Ollama counts the weights
    /// of CPU-side layers, their KV cache and the host compute buffers here;
    /// a model wholly on the GPUs reports none.
    pub fn cpu_bytes(&self) -> u64 {
        self.size.saturating_sub(self.size_vram)
    }

    /// The PROCESSOR column of `ollama ps`: `100% GPU`, `100% CPU` or
    /// `38%/62% CPU/GPU`.
    pub fn processor(&self) -> String {
        if self.size == 0 {
            return String::new();
        }
        if self.size_vram == 0 {
            return "100% CPU".into();
        }
        if self.size_vram >= self.size {
            return "100% GPU".into();
        }
        let cpu = (self.cpu_bytes() as f64 / self.size as f64 * 100.0).round() as u32;
        format!("{cpu}%/{}% CPU/GPU", 100 - cpu)
    }
}

/// What the daemon said about the model a slot shows.
#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    /// `None`: the daemon answered and does not have this model loaded.
    pub loaded: Option<PsModel>,
    /// Models the daemon has loaded besides this one.
    pub others: usize,
    /// Whether the slot's own port serves token counters (`/slots`). Only
    /// the llama-server runner does; the daemon and `ollama runner` do not.
    pub counters: bool,
}

/// `/api/ps` → loaded models, or `None` when the body is not that document.
pub fn parse_ps(body: &str) -> Option<Vec<PsModel>> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let models = v.get("models")?.as_array()?;
    let s = |m: &serde_json::Value, k: &str| {
        m.get(k)
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string()
    };
    Some(
        models
            .iter()
            .map(|m| {
                let details = m.get("details").cloned().unwrap_or_default();
                PsModel {
                    name: if s(m, "name").is_empty() {
                        s(m, "model")
                    } else {
                        s(m, "name")
                    },
                    size: m.get("size").and_then(|x| x.as_u64()).unwrap_or(0),
                    size_vram: m.get("size_vram").and_then(|x| x.as_u64()).unwrap_or(0),
                    context_length: m
                        .get("context_length")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0) as usize,
                    expires_at: parse_time(&s(m, "expires_at")),
                    family: s(&details, "family"),
                    parameter_size: s(&details, "parameter_size"),
                    quantization: s(&details, "quantization_level"),
                }
            })
            .collect(),
    )
}

fn parse_time(s: &str) -> Option<i64> {
    let t = chrono::DateTime::parse_from_rfc3339(s).ok()?.timestamp();
    // Go's zero time.Time, which a model still loading can carry.
    (t > 0).then_some(t)
}

/// Where a slot's daemon is, and which `/api/ps` entry is its model.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    pub host: String,
    pub port: u16,
    /// Tags whose weights are the slot's blob. Empty for the daemon itself,
    /// which shows whatever it has loaded first.
    pub names: Vec<String>,
    pub counters: bool,
}

impl Link {
    /// The daemon behind a detected server, if it is an Ollama one.
    pub fn for_model(m: &DetectedModel) -> Option<Link> {
        let blob_names = || m.path.as_deref().map(names_for_blob).unwrap_or_default();
        if m.engine == "llama.cpp" && is_runner(m) {
            let (host, port) = daemon_addr(m.pid);
            return Some(Link {
                host,
                port,
                names: blob_names(),
                counters: true,
            });
        }
        if m.engine != "ollama" {
            return None;
        }
        if is_legacy_runner(&m.cmdline) {
            // Its `--port` is the runner's own; the daemon is elsewhere.
            let (host, port) = daemon_addr(m.pid);
            return Some(Link {
                host,
                port,
                names: blob_names(),
                counters: false,
            });
        }
        Some(Link {
            host: m.host.clone(),
            port: m.port.unwrap_or(DEFAULT_PORT),
            names: Vec::new(),
            counters: false,
        })
    }

    /// This slot's entry among what the daemon has loaded. `None` when that
    /// cannot be told: a runner whose tag is unknown (its store's manifests
    /// unreadable) next to other loaded models.
    pub fn pick(&self, models: &[PsModel]) -> Option<Status> {
        let found = if !self.names.is_empty() {
            models
                .iter()
                .find(|p| self.names.iter().any(|n| same_name(n, &p.name)))
        } else if !self.counters || models.len() == 1 {
            models.first()
        } else {
            return None;
        };
        Some(Status {
            others: models.len() - usize::from(found.is_some()),
            loaded: found.cloned(),
            counters: self.counters,
        })
    }

    /// `None` when the daemon did not answer: an unreachable daemon is not
    /// one that has unloaded the model.
    pub async fn poll(&self, auth: &HttpAuth) -> Option<Status> {
        let body = http_get(&self.host, self.port, "/api/ps", auth)
            .await
            .ok()?;
        parse_ps(&body).and_then(|models| self.pick(&models))
    }
}

/// `llama3` and `llama3:latest` are one tag.
fn same_name(a: &str, b: &str) -> bool {
    let norm = |s: &str| s.strip_suffix(":latest").unwrap_or(s).to_string();
    norm(a) == norm(b)
}

/// A file in an Ollama store: `…/blobs/sha256-<hex>`.
pub fn is_blob(path: &Path) -> bool {
    let named = path
        .file_name()
        .and_then(|f| f.to_str())
        .is_some_and(|f| f.starts_with("sha256-"));
    let in_blobs = path
        .parent()
        .and_then(|p| p.file_name())
        .is_some_and(|d| d == "blobs");
    named && in_blobs
}

/// A llama-server that Ollama started: its binary sits in `lib/ollama`, or
/// it serves a blob from an Ollama store.
pub fn is_runner(m: &DetectedModel) -> bool {
    let argv0 = m
        .cmdline
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .replace('\\', "/");
    argv0.contains("lib/ollama/") || m.path.as_deref().is_some_and(is_blob)
}

/// `ollama runner --model … --port …`, the subprocess releases before the
/// llama-server one used.
pub fn is_legacy_runner(cmdline: &str) -> bool {
    cmdline.split_whitespace().nth(1) == Some("runner")
}

/// The tags whose manifest names this blob as its weights, e.g.
/// `qwen3:8b`, sorted. Empty when the store has no manifests to read.
pub fn names_for_blob(blob: &Path) -> Vec<String> {
    let Some(digest) = blob
        .file_name()
        .and_then(|f| f.to_str())
        .and_then(|f| f.strip_prefix("sha256-"))
        .map(|hex| format!("sha256:{hex}"))
    else {
        return Vec::new();
    };
    let Some(root) = blob.parent().and_then(Path::parent) else {
        return Vec::new();
    };
    let manifests = root.join("manifests");
    let mut names = Vec::new();
    walk_manifests(&manifests, &mut Vec::new(), &mut |parts, path| {
        if model_digest(path).as_deref() == Some(digest.as_str()) {
            if let Some(n) = display_name(parts) {
                names.push(n);
            }
        }
    });
    names.sort();
    names
}

/// `manifests/<registry>/<namespace>/<model>/<tag>`: four levels, no more.
fn walk_manifests(dir: &Path, parts: &mut Vec<String>, f: &mut dyn FnMut(&[String], &Path)) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in rd.flatten() {
        let path = ent.path();
        parts.push(ent.file_name().to_string_lossy().into_owned());
        if parts.len() < 4 && path.is_dir() {
            walk_manifests(&path, parts, f);
        } else if parts.len() == 4 && path.is_file() {
            f(parts, &path);
        }
        parts.pop();
    }
}

fn model_digest(manifest: &Path) -> Option<String> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(manifest).ok()?).ok()?;
    v.get("layers")?
        .as_array()?
        .iter()
        .find(|l| l.get("mediaType").and_then(|t| t.as_str()) == Some(MODEL_LAYER))?
        .get("digest")?
        .as_str()
        .map(str::to_string)
}

/// The name Ollama itself prints: the default registry and namespace left
/// out, the tag kept.
fn display_name(parts: &[String]) -> Option<String> {
    let [registry, namespace, model, tag] = parts else {
        return None;
    };
    Some(
        if registry == DEFAULT_REGISTRY && namespace == DEFAULT_NAMESPACE {
            format!("{model}:{tag}")
        } else if registry == DEFAULT_REGISTRY {
            format!("{namespace}/{model}:{tag}")
        } else {
            format!("{registry}/{namespace}/{model}:{tag}")
        },
    )
}

/// The weights of a tag, looked up in the local stores. For a daemon slot,
/// which has no runner command line to name the blob.
pub fn blob_for_name(name: &str) -> Option<PathBuf> {
    let (base, tag) = match name.rsplit_once(':') {
        Some((b, t)) if !t.contains('/') => (b, t),
        _ => (name, "latest"),
    };
    let parts: Vec<&str> = base.split('/').collect();
    let (registry, namespace, model) = match parts.as_slice() {
        [m] => (DEFAULT_REGISTRY, DEFAULT_NAMESPACE, *m),
        [n, m] => (DEFAULT_REGISTRY, *n, *m),
        [r, n, m] => (*r, *n, *m),
        _ => return None,
    };
    model_dirs().into_iter().find_map(|root| {
        let manifest = root
            .join("manifests")
            .join(registry)
            .join(namespace)
            .join(model)
            .join(tag);
        let digest = model_digest(&manifest)?;
        let blob = root
            .join("blobs")
            .join(digest.replacen("sha256:", "sha256-", 1));
        blob.is_file().then_some(blob)
    })
}

/// Where Ollama keeps models: `OLLAMA_MODELS`, the user's own store, and
/// the Linux service account's.
fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("OLLAMA_MODELS") {
        dirs.push(PathBuf::from(d));
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        dirs.push(PathBuf::from(home).join(".ollama").join("models"));
    }
    if cfg!(target_os = "linux") {
        dirs.push(PathBuf::from("/usr/share/ollama/.ollama/models"));
    }
    dirs
}

/// `OLLAMA_HOST` as the daemon reads it: an optional scheme, a host and a
/// port, any of which may be missing. Wildcard binds are dialled on loopback.
pub fn parse_host(raw: &str) -> Option<(String, u16)> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let (default_port, rest) = match raw.split_once("://") {
        Some(("http", r)) => (80, r),
        Some(("https", r)) => (443, r),
        Some((_, r)) => (DEFAULT_PORT, r),
        None => (DEFAULT_PORT, raw),
    };
    let hostport = rest.split('/').next().unwrap_or_default();
    let (host, port) = if let Some(v6) = hostport.strip_prefix('[') {
        let (h, tail) = v6.split_once(']')?;
        let port = tail.strip_prefix(':').and_then(|p| p.parse().ok());
        (h.to_string(), port)
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) if !h.contains(':') => (h.to_string(), p.parse().ok()),
            _ => (hostport.to_string(), None),
        }
    };
    Some((
        crate::model_detect::connect_host(&host),
        port.unwrap_or(default_port),
    ))
}

/// The daemon's address from `OLLAMA_HOST` in the environment of the
/// process or its parent (the daemon itself), else the default. The
/// environment of another user's process is unreadable, so a system
/// service is usually found by the default.
fn daemon_addr(pid: u32) -> (String, u16) {
    [Some(pid), parent_pid(pid)]
        .into_iter()
        .flatten()
        .find_map(env_host)
        .unwrap_or_else(|| ("127.0.0.1".into(), DEFAULT_PORT))
}

/// `OLLAMA_HOST` from a process's environment.
pub fn env_host(pid: u32) -> Option<(String, u16)> {
    crate::vision::read_environ(pid)
        .get("OLLAMA_HOST")
        .and_then(|v| parse_host(v))
}

#[cfg(target_os = "linux")]
fn parent_pid(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name is in parentheses and may hold spaces.
    let rest = stat.rsplit_once(')')?.1;
    rest.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(not(target_os = "linux"))]
fn parent_pid(_pid: u32) -> Option<u32> {
    None
}

/// When the model will be unloaded, for the header: `unloads in 4m`,
/// `kept loaded`, `unloading`. `None` while a request runs, since Ollama
/// only restarts the timer when it ends.
pub fn expiry_label(p: &PsModel, now: i64, busy: bool) -> Option<String> {
    let left = p.expires_at? - now;
    if busy {
        return None;
    }
    Some(if left >= FOREVER_SECS {
        "kept loaded".into()
    } else if left <= 0 {
        "unloading".into()
    } else {
        format!("unloads in {}", fmt_left(left))
    })
}

/// The same in at most three characters, for the models strip: `4m`, `∞`.
pub fn expiry_short(p: &PsModel, now: i64) -> Option<String> {
    let left = p.expires_at? - now;
    Some(if left >= FOREVER_SECS {
        "∞".into()
    } else {
        fmt_left(left.max(0))
    })
}

fn fmt_left(secs: i64) -> String {
    if secs >= 86_400 {
        format!("{}d", secs / 86_400)
    } else if secs >= 3600 {
        format!("{}h", secs / 3600)
    } else if secs >= 60 {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PS: &str = include_str!("../fixtures/ollama-ps.json");
    const PS_EMPTY: &str = include_str!("../fixtures/ollama-ps-empty.json");

    fn link(names: &[&str], counters: bool) -> Link {
        Link {
            host: "127.0.0.1".into(),
            port: DEFAULT_PORT,
            names: names.iter().map(|s| s.to_string()).collect(),
            counters,
        }
    }

    #[test]
    fn parses_a_captured_ps_document() {
        let models = parse_ps(PS).expect("ps");
        assert_eq!(models.len(), 2);
        let g = &models[0];
        assert_eq!(g.name, "granite3-moe:1b");
        assert_eq!(g.size, 956_626_369);
        assert_eq!(g.size_vram, 0);
        assert_eq!(g.context_length, 2048);
        assert_eq!(g.family, "granitemoe");
        assert_eq!(g.parameter_size, "1.3B");
        assert_eq!(g.quantization, "Q4_K_M");
        assert_eq!(g.expires_at, Some(1_792_755_727));
        assert_eq!(g.processor(), "100% CPU");
        // The API reference's example: wholly on the GPU, an offset time.
        let m = &models[1];
        assert_eq!(m.name, "mistral:latest");
        assert_eq!(m.processor(), "100% GPU");
        assert_eq!(m.cpu_bytes(), 0);
        assert_eq!(m.expires_at, Some(1_717_537_111));
        assert_eq!(parse_ps(PS_EMPTY), Some(Vec::new()));
        assert_eq!(parse_ps("404 page not found"), None);
    }

    #[test]
    fn a_split_model_reads_as_ollama_ps_prints_it() {
        let p = PsModel {
            size: 10_000,
            size_vram: 6_200,
            ..Default::default()
        };
        assert_eq!(p.cpu_bytes(), 3_800);
        assert_eq!(p.processor(), "38%/62% CPU/GPU");
    }

    #[test]
    fn a_slot_finds_its_own_tag_and_counts_the_rest() {
        let models = parse_ps(PS).unwrap();
        let s = link(&["mistral"], true).pick(&models).unwrap();
        assert_eq!(
            s.loaded.as_ref().map(|p| p.name.as_str()),
            Some("mistral:latest")
        );
        assert_eq!(s.others, 1);
        assert!(s.counters);
        // Its tag no longer listed: unloaded, not unknown.
        let gone = link(&["qwen3:8b"], true).pick(&models).unwrap();
        assert_eq!(gone.loaded, None);
        assert_eq!(gone.others, 2);
        // The daemon's own slot shows the first, which Ollama sorts by the
        // longest time left.
        let d = link(&[], false).pick(&models).unwrap();
        assert_eq!(d.loaded.unwrap().name, "granite3-moe:1b");
        assert_eq!(d.others, 1);
        assert_eq!(link(&[], false).pick(&[]).unwrap().loaded, None);
        // A runner that cannot name its tag takes the only model, and
        // guesses at none of several.
        assert!(link(&[], true).pick(&models).is_none());
        assert!(link(&[], true).pick(&models[..1]).unwrap().loaded.is_some());
    }

    #[test]
    fn expiry_reads_as_a_countdown() {
        let p = PsModel {
            expires_at: Some(1_000_000),
            ..Default::default()
        };
        let at = |now| expiry_label(&p, now, false);
        assert_eq!(at(1_000_000 - 250).as_deref(), Some("unloads in 4m"));
        assert_eq!(at(1_000_000 - 30).as_deref(), Some("unloads in 30s"));
        assert_eq!(at(1_000_000 - 7200).as_deref(), Some("unloads in 2h"));
        assert_eq!(at(1_000_001).as_deref(), Some("unloading"));
        assert_eq!(expiry_label(&p, 0, true), None);
        assert_eq!(expiry_short(&p, 1_000_000 - 250).as_deref(), Some("4m"));
        // keep_alive -1: Ollama reports a time ~292 years out.
        let forever = PsModel {
            expires_at: parse_time("2318-11-28T03:28:46.123456789Z"),
            ..Default::default()
        };
        assert_eq!(
            expiry_label(&forever, 1_792_755_727, false).as_deref(),
            Some("kept loaded")
        );
        assert_eq!(expiry_short(&forever, 1_792_755_727).as_deref(), Some("∞"));
        assert_eq!(parse_time("0001-01-01T00:00:00Z"), None);
    }

    #[test]
    fn ollama_host_is_read_as_the_daemon_reads_it() {
        let h = |s| parse_host(s);
        assert_eq!(h("0.0.0.0:11434"), Some(("127.0.0.1".into(), 11434)));
        assert_eq!(h("127.0.0.1:11999"), Some(("127.0.0.1".into(), 11999)));
        assert_eq!(h("192.168.1.5"), Some(("192.168.1.5".into(), 11434)));
        assert_eq!(h(":8000"), Some(("127.0.0.1".into(), 8000)));
        assert_eq!(h("http://10.0.0.2"), Some(("10.0.0.2".into(), 80)));
        assert_eq!(h("http://box:9000/"), Some(("box".into(), 9000)));
        assert_eq!(h("[::]:11434"), Some(("127.0.0.1".into(), 11434)));
        assert_eq!(h("[fe80::1]"), Some(("fe80::1".into(), 11434)));
        assert_eq!(h(""), None);
    }

    #[test]
    fn a_blob_is_named_by_the_manifests_that_point_at_it() {
        let root = std::env::temp_dir().join(format!("lv-ollama-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let hex = "6bd48eb2b42c83802e8b07a4a8d74cb15b641cf083d7d56f32974b27d64a3ac9";
        let manifest = format!(
            r#"{{"schemaVersion":2,"layers":[
                {{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:{hex}","size":821844672}},
                {{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:604785e6","size":664}}]}}"#
        );
        let tag = |parts: &[&str], body: &str| {
            let p = parts
                .iter()
                .fold(root.join("manifests"), |acc, part| acc.join(part));
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        };
        tag(
            &["registry.ollama.ai", "library", "granite3-moe", "1b"],
            &manifest,
        );
        tag(
            &["registry.ollama.ai", "someone", "granite", "latest"],
            &manifest,
        );
        tag(&["hf.co", "org", "repo", "Q4_K_M"], &manifest);
        tag(
            &["registry.ollama.ai", "library", "other", "latest"],
            r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:ffff"}]}"#,
        );
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        let blob = root.join("blobs").join(format!("sha256-{hex}"));
        std::fs::write(&blob, b"GGUF").unwrap();

        assert!(is_blob(&blob));
        assert!(!is_blob(Path::new("/m/sha256-abc.gguf.d/x")));
        assert_eq!(
            names_for_blob(&blob),
            vec![
                "granite3-moe:1b".to_string(),
                "hf.co/org/repo:Q4_K_M".to_string(),
                "someone/granite:latest".to_string(),
            ]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_runner_is_told_from_its_command_line() {
        assert!(is_legacy_runner(
            "/usr/local/bin/ollama runner --ollama-engine --model /m/blobs/sha256-ab --port 41234"
        ));
        assert!(!is_legacy_runner("/usr/local/bin/ollama serve"));
    }
}
