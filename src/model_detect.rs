use std::path::{Path, PathBuf};

use crate::gguf::{self, GgufInfo};

/// Information about a detected running LLM process
#[derive(Debug, Clone)]
pub struct DetectedModel {
    pub name: String,
    pub path: Option<PathBuf>,
    pub pid: u32,
    #[allow(dead_code)]
    pub process_name: String,
    pub engine: String,
    pub gpu_indices: Vec<u32>,
    pub mem_used_mb: u64,
    pub port: Option<u16>,
    pub ctx_max: Option<usize>,
    pub spec_type: Option<String>,
    #[allow(dead_code)]
    pub n_gpu_layers: Option<u32>,
    pub tensor_split: Vec<f32>,
    pub cmdline: String,
    pub gguf: Option<GgufInfo>,
    /// Weight byte layout from the tensor table (for bandwidth estimates).
    pub tensors: Option<gguf::TensorSummary>,
}

impl std::fmt::Display for DetectedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpus = if self.gpu_indices.is_empty() {
            "?".into()
        } else {
            self.gpu_indices
                .iter()
                .map(|g| g.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        write!(
            f,
            "{} (PID {} · {} · GPU {} · {} MB)",
            self.name, self.pid, self.engine, gpus, self.mem_used_mb
        )
    }
}

impl DetectedModel {
    /// Stable identity for routing poller samples back to a slot.
    pub fn key(&self) -> u32 {
        self.pid
    }

    /// Short label for compact multi-model panels: the alias or file stem,
    /// trimmed of the quant/size noise that makes every name look the same.
    pub fn short_name(&self) -> String {
        let n = self.name.trim();
        let n = n.strip_prefix("models--").unwrap_or(n);
        let n = n.split('/').next_back().unwrap_or(n);
        // Drop a trailing quant tag so "Qwen3-8B-UD-Q4_K_XL" reads "Qwen3-8B".
        let mut parts: Vec<&str> = n.split('-').collect();
        while parts.len() > 1 {
            let last = parts[parts.len() - 1].to_ascii_uppercase();
            let quantish = last.starts_with('Q') && last.chars().any(|c| c.is_ascii_digit());
            if quantish
                || matches!(
                    last.as_str(),
                    "UD" | "GGUF" | "K" | "XL" | "M" | "S" | "L" | "0" | "1"
                )
            {
                parts.pop();
            } else {
                break;
            }
        }
        parts.join("-")
    }

    /// Whether this looks like a bare daemon with nothing loaded.
    fn is_idle_daemon(&self) -> bool {
        self.mem_used_mb == 0 && self.path.is_none() && self.gguf.is_none()
    }

    pub fn n_layers(&self) -> usize {
        self.gguf.as_ref().map(|g| g.n_layers).unwrap_or(0)
    }

    pub fn n_heads(&self) -> usize {
        self.gguf.as_ref().map(|g| g.n_heads).unwrap_or(0)
    }

    pub fn n_experts(&self) -> usize {
        self.gguf.as_ref().map(|g| g.n_experts).unwrap_or(0)
    }

    pub fn n_experts_used(&self) -> usize {
        self.gguf.as_ref().map(|g| g.n_experts_used).unwrap_or(0)
    }
}

/// Populate model identity and tensor metadata after a model path is learned
/// from somewhere other than the process command line (for example llama.cpp
/// `/props` when the server was launched with `-hf`).
pub fn load_gguf_metadata(model: &mut DetectedModel, path: PathBuf) {
    let path = if !path.exists() {
        resolve_container_path(model.pid, &path).unwrap_or(path)
    } else {
        path
    };
    model.path = Some(path.clone());
    if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
        return;
    }
    if let Ok(info) = gguf::read_info(&path) {
        if model.name.starts_with('[') || model.name == "llama-server" || model.name.is_empty() {
            model.name = info.name.clone();
        }
        model.gguf = Some(info);
        model.tensors = gguf::read_tensor_summary(&path).ok();
    }
}

/// Undo the escapes mountinfo applies to mount points/roots
/// (\040 space, \011 tab, \013 newline).
fn unescape_mount(s: &str) -> String {
    s.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\013", "\n")
}

/// A containerized server reports its model path in its own mount
/// namespace (e.g. `/model`); that path doesn't exist on the host, so file
/// sizes and GGUF metadata would read nothing. Re-anchor it through
/// /proc/<pid>/mountinfo: mount device numbers are global (one kernel), so
/// the device of the mount containing the path identifies the host-side
/// mount, whose mount point is the host prefix.
fn resolve_container_path(pid: u32, path: &Path) -> Option<PathBuf> {
    let target = path.to_string_lossy().to_string();
    let info = std::fs::read_to_string(format!("/proc/{pid}/mountinfo")).ok()?;

    // The deepest mount in the process's table whose mount point is a
    // prefix of the target path.
    let mut best: Option<(String, String, String)> = None;
    for line in info.lines() {
        let f: Vec<&str> = line.splitn(6, ' ').collect();
        if f.len() < 6 {
            continue;
        }
        let mp = unescape_mount(f[4]);
        if target.starts_with(&mp)
            && best
                .as_ref()
                .map_or(true, |(_, _, bmp)| mp.len() > bmp.len())
        {
            best = Some((f[2].to_string(), unescape_mount(f[3]), mp));
        }
    }
    let (dev, root, mp) = best?;

    // Find that device in our own table: its mount point is the host
    // prefix of the mount root.
    let self_info = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let mut host_prefix: Option<String> = None;
    for line in self_info.lines() {
        let f: Vec<&str> = line.splitn(6, ' ').collect();
        if f.len() >= 6 && f[2] == dev {
            let hmp = unescape_mount(f[4]);
            if host_prefix.as_ref().map_or(true, |p| hmp.len() > p.len()) {
                host_prefix = Some(hmp);
            }
        }
    }
    let host_prefix = host_prefix?;

    let suffix = target.strip_prefix(&mp).unwrap_or("");
    let host = format!("{host_prefix}{root}{suffix}");
    let host = host.trim_end_matches('/');
    let p = if host.is_empty() {
        PathBuf::from("/")
    } else {
        PathBuf::from(host)
    };
    // Only trust a re-anchored path that actually resolves on the host.
    p.exists().then_some(p)
}

/// Architecture fields from a HuggingFace-style `config.json`. Looks
/// under `text_config` when present (nested Qwen/Llama configs). Fills
/// the same struct as the GGUF header so the layers and experts panels
/// work for vLLM and SGLang safetensors dirs.
fn hf_config_info(dir: &Path) -> Option<GgufInfo> {
    let txt = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    let root = v.get("text_config").unwrap_or(&v);
    let usize_at =
        |obj: &serde_json::Value, k: &str| obj.get(k).and_then(|x| x.as_u64()).map(|n| n as usize);
    let pick = |k: &str| usize_at(root, k).or_else(|| usize_at(&v, k));
    let n_layers = pick("num_hidden_layers").unwrap_or(0);
    let n_heads = pick("num_attention_heads").unwrap_or(0);
    let n_kv_heads = pick("num_key_value_heads").unwrap_or(n_heads);
    let n_experts = pick("num_experts")
        .or_else(|| pick("num_local_experts"))
        .unwrap_or(0);
    let n_experts_used = pick("num_experts_per_tok").unwrap_or(0);
    let ctx_train = pick("max_position_embeddings").unwrap_or(0);
    let n_embd = pick("hidden_size").unwrap_or(0);
    let architecture = root
        .get("model_type")
        .or_else(|| v.get("model_type"))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let name = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    if n_layers == 0 && ctx_train == 0 && n_heads == 0 {
        return None;
    }
    Some(GgufInfo {
        name,
        architecture,
        n_layers,
        n_heads,
        n_kv_heads,
        n_experts,
        n_experts_used,
        ctx_train,
        n_embd,
        n_mtp: 0,
        engram: None,
    })
}

/// Whether a process that `looks_like_llm` flagged as vLLM is actually a
/// vLLM helper rather than the server itself. The main process forks
/// named workers (`VLLM::EngineCore`, `VLLM::Worker_TPn`) that inherit
/// the parent's command line via fork -- so they too "contain vllm" and
/// carry a `--port` -- and a `docker run` client or wrapper script that
/// mentions vllm in its arguments is not a server at all. Only the bare
/// `vllm` main process can legitimately omit `--port` (vLLM's default is
/// 8000).
fn is_vllm_phantom(process_name: &str, cmdline: &str) -> bool {
    if process_name.starts_with("VLLM::") || process_name.starts_with("docker") {
        return true;
    }
    // The real server is the `vllm` launcher or the python module form,
    // whatever its comm says (python3, pt_main_thread, or a full
    // executable path on the nvidia-smi side). Wrapper scripts and
    // clients merely mention vllm somewhere in their arguments.
    let argv0_is_vllm = cmdline
        .split_whitespace()
        .next()
        .map(|t| Path::new(t).file_name().is_some_and(|f| f == "vllm"))
        .unwrap_or(false);
    !(argv0_is_vllm || cmdline.contains("vllm.entrypoints"))
}

/// SGLang forks `sglang::scheduler` (holds the GPU memory) and
/// `sglang::detokenizer`. Only the launcher process serves HTTP, so
/// workers are folded into their parent rather than listed as models.
fn is_sglang_worker(process_name: &str) -> bool {
    let base = Path::new(process_name)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| process_name.to_string());
    base.starts_with("sglang::")
}

fn ppid_from_stat(txt: &str) -> Option<u32> {
    let rest = txt.rsplit_once(')')?.1;
    rest.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(target_os = "linux")]
fn parent_pid(pid: u32) -> Option<u32> {
    let txt = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    ppid_from_stat(&txt)
}

#[cfg(not(target_os = "linux"))]
fn parent_pid(_pid: u32) -> Option<u32> {
    None
}

/// A vLLM server in a Docker container reports its container-internal
/// port (`--port 8000` inside), but from the host it is reachable at the
/// published port (docker-proxy, e.g. `8003:8000`). This dashboard runs
/// in the host namespaces, so polling the container port would hit a
/// different service (or nothing). Re-anchor: if the server lives in a
/// different network namespace than we do, find the docker-proxy -- which
/// runs in our netns and carries `-container-ip` / `-host-port` in its
/// argv -- whose container IP appears in the server's namespace, and use
/// its host port. Without CAP_SYS_PTRACE the other namespace's tables are
/// unreadable; fall back to the command-line port in that case.
fn resolve_vllm_host_port(pid: u32, cmdline_port: Option<u16>) -> Option<u16> {
    resolve_container_host_port(pid, cmdline_port, 8000)
}

/// A containerized server reports its container-internal port, but from
/// the host it is reachable at the published port (docker-proxy). Re-anchor
/// when the process lives in a different network namespace.
fn resolve_container_host_port(
    pid: u32,
    cmdline_port: Option<u16>,
    default_port: u16,
) -> Option<u16> {
    let own = match std::fs::read_link("/proc/self/ns/net") {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => return cmdline_port,
    };
    let server = match std::fs::read_link(format!("/proc/{pid}/ns/net")) {
        Ok(p) => p.to_string_lossy().into_owned(),
        // Same namespace as us (or unreadable): the command-line port is
        // already a host port.
        Err(_) => return cmdline_port,
    };
    if server == own {
        return cmdline_port.or(Some(default_port));
    }
    let ips = netns_ipv4s(pid)?;
    if ips.is_empty() {
        return cmdline_port;
    }
    let dir = std::fs::read_dir("/proc").ok()?;
    for ent in dir.flatten() {
        let proxy_pid: u32 = match ent.file_name().to_string_lossy().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let comm = match std::fs::read_to_string(format!("/proc/{proxy_pid}/comm")) {
            Ok(c) => c.trim().to_string(),
            Err(_) => continue,
        };
        if comm != "docker-proxy" {
            continue;
        }
        let args: Vec<String> = match std::fs::read(format!("/proc/{proxy_pid}/cmdline")) {
            Ok(raw) if !raw.is_empty() => raw
                .split(|b| *b == 0)
                .filter(|p| !p.is_empty())
                .map(|p| String::from_utf8_lossy(p).into_owned())
                .collect(),
            _ => continue,
        };
        let mut container_ip: Option<String> = None;
        let mut container_port: Option<u16> = None;
        let mut host_port: Option<u16> = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "-container-ip" => {
                    if let Some(v) = args.get(i + 1) {
                        container_ip = Some(v.clone());
                        i += 1;
                    }
                }
                "-container-port" => {
                    if let Some(v) = args.get(i + 1) {
                        container_port = v.parse().ok();
                        i += 1;
                    }
                }
                "-host-port" => {
                    if let Some(v) = args.get(i + 1) {
                        host_port = v.parse().ok();
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        // A container can publish several ports (one docker-proxy each,
        // all sharing the container IP); only the mapping whose container
        // side is the server's own port is ours.
        if let (Some(ip), Some(port)) = (&container_ip, host_port) {
            if ips.iter().any(|x| x == ip)
                && container_port == Some(cmdline_port.unwrap_or(default_port))
            {
                return Some(port);
            }
        }
    }
    cmdline_port
}

/// The IPv4 addresses bound inside a process's network namespace, read
/// from its routing table (the `Local:` section of /proc/<pid>/net/
/// fib_trie).
fn netns_ipv4s(pid: u32) -> Option<Vec<String>> {
    let data = std::fs::read_to_string(format!("/proc/{pid}/net/fib_trie")).ok()?;
    Some(fib_local_ips(&data))
}

/// Parse the bound IPv4 addresses out of a fib_trie dump. Section
/// headers ("Local:", "Broadcast:", ...) sit at column 0; every other
/// line is indented. Under "Local:" each address is a two-line entry:
/// the bare address below a tree of `|`/`-`/`+` glyphs, then a line
/// starting with `/prefix` ("  /32 host LOCAL").
fn fib_local_ips(text: &str) -> Vec<String> {
    let mut section = "";
    let mut ips: Vec<String> = Vec::new();
    for line in text.lines() {
        if !line.starts_with(' ') && line.ends_with(':') {
            section = line;
            continue;
        }
        if section != "Local:" {
            continue;
        }
        // Drop the trie's tree glyphs; leaf lines then start with the
        // bare address, while subnet headers ("0.0.0.0/0 3 0 5") and the
        // "/32 host LOCAL" continuation lines fail the IPv4 check.
        let rest = line.trim_start_matches(|c: char| !c.is_ascii_alphanumeric());
        if rest.is_empty() {
            continue;
        }
        let first = rest.split(' ').next().unwrap_or("");
        if is_ipv4(first) {
            ips.push(first.to_string());
        }
    }
    ips
}

fn is_ipv4(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() == 4
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

/// Scan GPU compute apps + process cmdlines for inference servers.
/// Every server found is returned, best first; the caller decides how many
/// to monitor.
pub fn detect_models() -> Vec<DetectedModel> {
    let mut gpu_procs = nvidia_compute_apps();
    if gpu_procs.is_empty() {
        gpu_procs = amd_compute_apps();
    }
    let mut by_pid: std::collections::HashMap<u32, DetectedModel> =
        std::collections::HashMap::new();
    // SGLang workers hold the GPU memory; fold it onto the launcher PID.
    let mut worker_gpu: std::collections::HashMap<u32, (u64, Vec<u32>)> =
        std::collections::HashMap::new();

    for app in gpu_procs {
        let cmdline = read_cmdline(app.pid).unwrap_or_else(|| app.process_name.clone());
        if is_sglang_worker(&app.process_name) {
            if let Some(ppid) = parent_pid(app.pid) {
                let e = worker_gpu.entry(ppid).or_insert((0, Vec::new()));
                e.0 = e.0.saturating_add(app.mem_used_mb);
                if !e.1.contains(&app.gpu_index) {
                    e.1.push(app.gpu_index);
                }
            }
            continue;
        }
        if !looks_like_llm(&app.process_name, &cmdline) || is_self(app.pid, &cmdline) {
            continue;
        }
        let mut parsed = parse_cmdline(&app.process_name, &cmdline);
        if parsed.engine == "vllm" && is_vllm_phantom(&app.process_name, &cmdline) {
            continue;
        }
        if parsed.engine == "vllm" {
            parsed.port = resolve_vllm_host_port(app.pid, parsed.port);
        }
        if parsed.engine == "sglang" {
            parsed.port = resolve_container_host_port(app.pid, parsed.port, 30000);
        }
        let entry = by_pid.entry(app.pid).or_insert_with(|| DetectedModel {
            name: parsed.name.clone(),
            path: parsed.path.clone(),
            pid: app.pid,
            process_name: app.process_name.clone(),
            engine: parsed.engine.clone(),
            gpu_indices: Vec::new(),
            mem_used_mb: 0,
            port: parsed.port,
            ctx_max: parsed.ctx_max,
            spec_type: parsed.spec_type.clone(),
            n_gpu_layers: parsed.n_gpu_layers,
            tensor_split: parsed.tensor_split.clone(),
            cmdline: cmdline.clone(),
            gguf: None,
            tensors: None,
        });
        if !entry.gpu_indices.contains(&app.gpu_index) {
            entry.gpu_indices.push(app.gpu_index);
        }
        entry.mem_used_mb = entry.mem_used_mb.saturating_add(app.mem_used_mb);
    }

    // /proc scan for engines that might not appear in nvidia-smi
    for (pid, name, cmdline) in walk_proc_llms() {
        if by_pid.contains_key(&pid) {
            continue;
        }
        let mut parsed = parse_cmdline(&name, &cmdline);
        // vLLM forks named helpers (`VLLM::EngineCore`,
        // `VLLM::Worker_TPn`) that inherit the parent command line, and
        // `docker run` clients and wrapper scripts carry "vllm" in their
        // arguments; none of them serves an API. Drop them so the model
        // list is not full of duplicates of the real server.
        if parsed.engine == "vllm" && is_vllm_phantom(&name, &cmdline) {
            continue;
        }
        if is_sglang_worker(&name) {
            continue;
        }
        // A vLLM server in a container reports its internal port;
        // re-anchor to the port published on the host.
        if parsed.engine == "vllm" {
            parsed.port = resolve_vllm_host_port(pid, parsed.port);
        }
        if parsed.engine == "sglang" {
            parsed.port = resolve_container_host_port(pid, parsed.port, 30000);
        }
        by_pid.insert(
            pid,
            DetectedModel {
                name: parsed.name,
                path: parsed.path,
                pid,
                process_name: name,
                engine: parsed.engine,
                gpu_indices: Vec::new(),
                mem_used_mb: 0,
                port: parsed.port,
                ctx_max: parsed.ctx_max,
                spec_type: parsed.spec_type,
                n_gpu_layers: parsed.n_gpu_layers,
                tensor_split: parsed.tensor_split,
                cmdline,
                gguf: None,
                tensors: None,
            },
        );
    }

    for (ppid, (mem, gpus)) in worker_gpu {
        if let Some(m) = by_pid.get_mut(&ppid) {
            m.mem_used_mb = m.mem_used_mb.saturating_add(mem);
            for g in gpus {
                if !m.gpu_indices.contains(&g) {
                    m.gpu_indices.push(g);
                }
            }
        }
    }

    let mut models: Vec<DetectedModel> = by_pid.into_values().collect();
    for m in &mut models {
        if let Some(path) = m.path.clone() {
            // A containerized server reports the model path as seen from
            // its own mount namespace (e.g. /model); re-anchor it through
            // /proc/<pid>/mountinfo so the size (and the GGUF metadata
            // below) reads the host-side file.
            let path = if !path.exists() {
                resolve_container_path(m.pid, &path).unwrap_or(path)
            } else {
                path
            };
            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                load_gguf_metadata(m, path);
            } else if path.is_dir() {
                // A safetensors dir has no GGUF header; read config.json
                // so layers/heads/experts populate the same panels. vLLM
                // and SGLang default context is max_position_embeddings
                // when --max-model-len / --context-length is absent.
                if let Some(info) = hf_config_info(&path) {
                    if m.ctx_max.is_none() && info.ctx_train > 0 {
                        m.ctx_max = Some(info.ctx_train);
                    }
                    if m.gguf.is_none() {
                        m.gguf = Some(info);
                    }
                }
            }
        }
        m.gpu_indices.sort_unstable();
    }

    // GPU memory first; when nvidia-smi is unavailable that is zero for all,
    // so fall back to "has a model loaded on its command line" and then to a
    // serving engine over a resident daemon (ollama with nothing loaded).
    models.sort_by_key(|m| {
        let has_model = m.path.is_some() || m.gguf.is_some();
        let engine_rank = match m.engine.as_str() {
            "llama.cpp" | "vllm" | "sglang" | "exllamav2" => 2,
            "ollama" => 0,
            _ => 1,
        };
        std::cmp::Reverse((m.mem_used_mb, has_model, engine_rank))
    });
    // An engine daemon with nothing loaded (ollama waiting for a request) is
    // noise next to a server that is actually serving a model.
    if models.iter().any(|m| !m.is_idle_daemon()) {
        models.retain(|m| !m.is_idle_daemon());
    }
    models
}

/// This dashboard is itself a process with `--model` on its command line;
/// without this it would list itself as a running LLM.
fn is_self(pid: u32, cmdline: &str) -> bool {
    pid == std::process::id() || cmdline.contains("llm-visuals")
}

#[derive(Default)]
struct ParsedCmd {
    name: String,
    path: Option<PathBuf>,
    engine: String,
    port: Option<u16>,
    ctx_max: Option<usize>,
    spec_type: Option<String>,
    n_gpu_layers: Option<u32>,
    tensor_split: Vec<f32>,
}

fn looks_like_llm(process_name: &str, cmdline: &str) -> bool {
    let p = process_name.to_lowercase();
    let c = cmdline.to_lowercase();
    let keys = [
        "llama-server",
        "llama-cli",
        "llama.cpp",
        "ollama",
        "vllm",
        "sglang",
        "exllama",
        "text-generation",
        "aphrodite",
        "tensorrt-llm",
        "lmdeploy",
        "kobold",
        "tabbyapi",
        "transformers",
    ];
    // A bare ".gguf" substring match is too loose: it fires on anything that
    // merely mentions a GGUF filename (a download, an `ls`, a `cp`), not just
    // a server loading one. `names_a_model` already validates that a
    // `--model`/`-m` flag actually points at a model file.
    //
    // The keyword search itself must not be a raw substring scan over the
    // whole command line either: a `bash -c '...pgrep -f "llama-server.*
    // --port 8090"...'` monitoring loop, or a path segment like
    // `run_vllm.sh`, contains these keywords without being the server. Match
    // only a whole argv token (or its path basename), the shape an actual
    // invocation takes.
    let token_matches = |t: &str| {
        keys.iter()
            .any(|k| t == *k || Path::new(t).file_name().is_some_and(|f| f == *k))
            // `python -m sglang.launch_server` is one argv token that is
            // not the bare keyword `sglang`.
            || t.contains("sglang.launch_server")
            || t.contains("sglang.srt.entrypoints")
    };
    keys.iter().any(|k| p.contains(k))
        || c.split_whitespace().any(token_matches)
        || names_a_model(cmdline)
}

/// Interpreters whose `-m` means "run this module", not "load this model".
/// Without this, `python3 -m http.server` and `gjs -m …/org.gnome.Shell.js`
/// both read as inference servers.
fn is_interpreter(argv0: &str) -> bool {
    let base = Path::new(argv0)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| argv0.to_string())
        .to_lowercase();
    let base = base.trim_end_matches(".exe").to_string();
    if base.starts_with("python") {
        return true;
    }
    matches!(
        base.as_str(),
        "gjs"
            | "node"
            | "nodejs"
            | "ruby"
            | "perl"
            | "bash"
            | "sh"
            | "zsh"
            | "java"
            | "dotnet"
            | "uv"
            | "uvx"
    )
}

/// Whether a `--model` / `-m` argument points at something that is plausibly
/// a model: a weights file, a path that exists, or a HuggingFace-style id.
fn is_model_ref(v: &str) -> bool {
    let lower = v.to_lowercase();
    if [".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx"]
        .iter()
        .any(|e| lower.ends_with(e))
    {
        return true;
    }
    if v.starts_with('/') || v.starts_with('.') || v.starts_with('~') {
        // A bare directory of weights counts; a random file on disk does not.
        return Path::new(v).is_dir();
    }
    // "org/name", the HuggingFace form.
    v.matches('/').count() == 1 && !v.contains('.') && v.len() > 3
}

/// Last-resort match for engines not in the keyword list: a command line that
/// actually names a model file.
fn names_a_model(cmdline: &str) -> bool {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    let interpreter = toks.first().map(|t| is_interpreter(t)).unwrap_or(false);
    for (i, t) in toks.iter().enumerate() {
        let value = if let Some(v) = t.strip_prefix("--model=") {
            Some(v)
        } else if let Some(v) = t.strip_prefix("--model-path=") {
            Some(v)
        } else if matches!(*t, "--model" | "--model-path") || (*t == "-m" && !interpreter) {
            toks.get(i + 1).copied()
        } else {
            None
        };
        if value.map(is_model_ref).unwrap_or(false) {
            return true;
        }
    }
    false
}

fn engine_from(process_name: &str, cmdline: &str) -> String {
    let blob = format!("{process_name} {cmdline}").to_lowercase();
    if blob.contains("llama-server") || blob.contains("llama.cpp") {
        "llama.cpp".into()
    } else if blob.contains("ollama") {
        "ollama".into()
    } else if blob.contains("vllm") {
        "vllm".into()
    } else if blob.contains("sglang") {
        "sglang".into()
    } else if blob.contains("exllama") {
        "exllamav2".into()
    } else if blob.contains("ollama") {
        "ollama".into()
    } else {
        "llm".into()
    }
}

fn parse_cmdline(process_name: &str, cmdline: &str) -> ParsedCmd {
    let tokens: Vec<&str> = cmdline.split_whitespace().collect();
    let mut parsed = ParsedCmd {
        engine: engine_from(process_name, cmdline),
        ..Default::default()
    };

    let mut i = 0;
    // Only tokens after `serve` / `api-server` can be vLLM's positional
    // weights path; without this the catch-all below eats argv[0] (the
    // launcher binary's own path) or a stray flag value.
    let mut seen_serve = false;
    while i < tokens.len() {
        let t = tokens[i];
        let (key, inline) = if let Some((k, v)) = t.split_once('=') {
            (k, Some(v))
        } else {
            (t, None)
        };
        let next = || {
            inline
                .map(|s| s.to_string())
                .or_else(|| tokens.get(i + 1).map(|s| s.to_string()))
        };

        match key {
            "--model" | "-m" | "--model-path" => {
                if let Some(v) = next() {
                    let p = PathBuf::from(&v);
                    parsed.name = p
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or(v.clone());
                    parsed.path = Some(p);
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--alias" => {
                if let Some(v) = next() {
                    parsed.name = v;
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--port" => {
                if let Some(v) = next() {
                    parsed.port = v.parse().ok();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--ctx-size" | "--ctx_size" | "-c" => {
                if let Some(v) = next() {
                    parsed.ctx_max = v.parse().ok();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            // SGLang's context-length flag (llama.cpp's is --ctx-size).
            "--context-length" | "--context_length" => {
                if let Some(v) = next() {
                    parsed.ctx_max = v.parse().ok();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            // vLLM's context-length flag (llama.cpp's is --ctx-size).
            "--max-model-len" | "--max_model_len" => {
                if let Some(v) = next() {
                    parsed.ctx_max = v.parse().ok();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            // vLLM's served-name flag: overrides both the displayed name
            // and the model_name label the metrics guard expects.
            "--served-model-name" | "--served_model_name" => {
                if let Some(v) = next() {
                    // vLLM accepts comma-separated names; the first is the
                    // primary identity.
                    parsed.name = v.split(',').next().unwrap_or(&v).to_string();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            // vLLM takes the weights path positionally after `serve` /
            // `api-server`; there is no --model key to capture it.
            "serve" | "api-server" if parsed.engine == "vllm" => {
                seen_serve = true;
                if let Some(v) = next().filter(|_| parsed.path.is_none()) {
                    if !v.starts_with('-') {
                        let p = PathBuf::from(&v);
                        if parsed.name.is_empty() {
                            parsed.name = p
                                .file_stem()
                                .map(|s| s.to_string_lossy().into_owned())
                                .unwrap_or(v.clone());
                        }
                        parsed.path = Some(p);
                        if inline.is_none() {
                            i += 1;
                        }
                    }
                }
            }
            "--spec-type"
            | "--spec_type"
            | "--speculative-algorithm"
            | "--speculative_algorithm" => {
                if let Some(v) = next() {
                    parsed.spec_type = Some(v);
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--n-gpu-layers" | "-ngl" => {
                if let Some(v) = next() {
                    parsed.n_gpu_layers = v.parse().ok();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--tensor-split" | "--tensor_split" => {
                if let Some(v) = next() {
                    parsed.tensor_split =
                        v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            // vLLM also takes the weights path as a bare positional even
            // later in the argument list; the named arms above only catch
            // the direct `serve <path>` form. Require a path-shaped token
            // (a '/' or a weights extension) so bare values of unhandled
            // flags ("--dtype float16") are not mistaken for the model.
            _ if parsed.engine == "vllm"
                && seen_serve
                && !key.starts_with('-')
                && parsed.path.is_none()
                && (key.contains('/')
                    || key.ends_with(".gguf")
                    || key.ends_with(".safetensors")) =>
            {
                let p = PathBuf::from(&key);
                if parsed.name.is_empty() {
                    parsed.name = p
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| key.to_string());
                }
                parsed.path = Some(p);
            }
            _ => {}
        }
        i += 1;
    }

    if parsed.name.is_empty() {
        parsed.name = Path::new(process_name)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| process_name.to_string());
    }
    if parsed.port.is_none() && parsed.engine == "llama.cpp" {
        parsed.port = Some(8080);
    }
    if parsed.port.is_none() && parsed.engine == "ollama" {
        parsed.port = Some(11434);
    }
    if parsed.port.is_none() && parsed.engine == "sglang" {
        parsed.port = Some(30000);
    }
    parsed
}

#[cfg(target_os = "linux")]
fn walk_proc_llms() -> Vec<(u32, String, String)> {
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return out;
    };
    for ent in dir.flatten() {
        let pid: u32 = match ent.file_name().to_string_lossy().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let cmdline = match read_cmdline(pid) {
            Some(c) if !c.is_empty() => c,
            _ => continue,
        };
        let name = std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .unwrap_or_default()
            .trim()
            .to_string();
        if looks_like_llm(&name, &cmdline) && !is_self(pid, &cmdline) {
            out.push((pid, name, cmdline));
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn read_cmdline(pid: u32) -> Option<String> {
    let raw = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    if raw.is_empty() {
        return None;
    }
    Some(
        raw.split(|b| *b == 0)
            .filter(|p| !p.is_empty())
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// No /proc here (Windows, macOS): list processes through sysinfo.
#[cfg(not(target_os = "linux"))]
fn walk_proc_llms() -> Vec<(u32, String, String)> {
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        ProcessesToUpdate::All,
        true,
        ProcessRefreshKind::nothing().with_cmd(UpdateKind::Always),
    );
    sys.processes()
        .iter()
        .filter_map(|(pid, p)| {
            let pid = pid.as_u32();
            let name = p.name().to_string_lossy().into_owned();
            // Another user's or an elevated process hides its argv; its name
            // still identifies the engine and the default port.
            let cmdline = cmdline_of(p).unwrap_or_else(|| name.clone());
            (looks_like_llm(&name, &cmdline) && !is_self(pid, &cmdline))
                .then_some((pid, name, cmdline))
        })
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn read_cmdline(pid: u32) -> Option<String> {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
    let pid = Pid::from_u32(pid);
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_cmd(UpdateKind::Always),
    );
    cmdline_of(sys.process(pid)?)
}

#[cfg(not(target_os = "linux"))]
fn cmdline_of(p: &sysinfo::Process) -> Option<String> {
    let args: Vec<String> = p
        .cmd()
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    (!args.is_empty()).then(|| args.join(" "))
}

struct ComputeApp {
    pid: u32,
    process_name: String,
    gpu_index: u32,
    mem_used_mb: u64,
}

fn gpu_uuid_index_map() -> std::collections::HashMap<String, u32> {
    let output = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return std::collections::HashMap::new(),
    };
    let mut map = std::collections::HashMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        if let Ok(idx) = parts[0].trim().parse::<u32>() {
            map.insert(parts[1].trim().to_string(), idx);
        }
    }
    map
}

fn nvidia_compute_apps() -> Vec<ComputeApp> {
    let uuid_map = gpu_uuid_index_map();
    let output = match std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let mut apps = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            continue;
        }
        let uuid = parts[0].trim();
        let pid = match parts[1].trim().parse::<u32>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let process_name = parts[2].trim().to_string();
        // Windows (WDDM) reports "[N/A]": keep the process, memory unknown.
        let mem_used = parts[3].trim().parse::<u64>().unwrap_or(0);
        let gpu_index = uuid_map.get(uuid).copied().unwrap_or(0);
        apps.push(ComputeApp {
            pid,
            process_name,
            gpu_index,
            mem_used_mb: mem_used,
        });
    }
    apps
}

#[derive(Debug, PartialEq, Eq)]
struct AmdClient {
    pdev: String,
    client_id: String,
    mem_used_kib: u64,
}

fn parse_amd_fdinfo(text: &str) -> Option<AmdClient> {
    let value = |key: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(key))
            .map(str::trim)
    };
    if value("drm-driver:")? != "amdgpu" {
        return None;
    }
    let mem_used_kib = value("drm-memory-vram:")?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(AmdClient {
        pdev: value("drm-pdev:")?.to_string(),
        client_id: value("drm-client-id:")?.to_string(),
        mem_used_kib,
    })
}

#[cfg(target_os = "linux")]
fn amd_compute_apps() -> Vec<ComputeApp> {
    use std::collections::HashMap;

    let mut cards: Vec<(String, PathBuf)> = std::fs::read_dir("/sys/class/drm")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let suffix = name.to_str()?.strip_prefix("card")?.to_string();
            if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let device = entry.path().join("device");
            let vendor = std::fs::read_to_string(device.join("vendor")).ok()?;
            (vendor.trim() == "0x1002").then_some((suffix, device))
        })
        .collect();
    cards.sort_by_key(|(card, _)| card.parse::<u32>().unwrap_or(u32::MAX));
    let pdev_to_index: HashMap<String, u32> = cards
        .into_iter()
        .enumerate()
        .filter_map(|(index, (_, device))| {
            let uevent = std::fs::read_to_string(device.join("uevent")).ok()?;
            let pdev = uevent
                .lines()
                .find_map(|line| line.strip_prefix("PCI_SLOT_NAME="))?;
            Some((pdev.to_string(), index as u32))
        })
        .collect();
    if pdev_to_index.is_empty() {
        return Vec::new();
    }

    let mut clients: HashMap<(u32, u32, String), u64> = HashMap::new();
    let Ok(proc_dir) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    for process in proc_dir.flatten() {
        let Ok(pid) = process.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        let Ok(fdinfo) = std::fs::read_dir(process.path().join("fdinfo")) else {
            continue;
        };
        for fd in fdinfo.flatten() {
            let Ok(text) = std::fs::read_to_string(fd.path()) else {
                continue;
            };
            let Some(client) = parse_amd_fdinfo(&text) else {
                continue;
            };
            let Some(&gpu_index) = pdev_to_index.get(&client.pdev) else {
                continue;
            };
            clients
                .entry((pid, gpu_index, client.client_id))
                .and_modify(|mem| *mem = (*mem).max(client.mem_used_kib))
                .or_insert(client.mem_used_kib);
        }
    }

    let mut per_process: HashMap<(u32, u32), u64> = HashMap::new();
    for ((pid, gpu_index, _), mem_kib) in clients {
        *per_process.entry((pid, gpu_index)).or_default() += mem_kib;
    }
    per_process
        .into_iter()
        .map(|((pid, gpu_index), mem_kib)| ComputeApp {
            pid,
            process_name: std::fs::read_to_string(format!("/proc/{pid}/comm"))
                .unwrap_or_default()
                .trim()
                .to_string(),
            gpu_index,
            mem_used_mb: mem_kib / 1024,
        })
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn amd_compute_apps() -> Vec<ComputeApp> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama_server_cmdline() {
        let cmd = "/home/dingo/models/llama.cpp/build/bin/llama-server --model /home/dingo/models/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf --alias qwen3.6-35b-a3b --host 0.0.0.0 --port 8080 --n-gpu-layers 99 --tensor-split 63,37 --spec-type draft-mtp --ctx-size 98304";
        let p = parse_cmdline("llama-server", cmd);
        assert_eq!(p.name, "qwen3.6-35b-a3b");
        assert_eq!(p.port, Some(8080));
        assert_eq!(p.ctx_max, Some(98304));
        assert_eq!(p.spec_type.as_deref(), Some("draft-mtp"));
        assert_eq!(p.tensor_split, vec![63.0, 37.0]);
        assert_eq!(p.engine, "llama.cpp");
        assert!(p
            .path
            .unwrap()
            .ends_with("Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf"));
    }

    #[test]
    fn parses_amd_drm_client_memory() {
        let text = "drm-driver:\tamdgpu\n\
                    drm-client-id:\t39\n\
                    drm-pdev:\t0000:43:00.0\n\
                    drm-memory-vram:\t15203992 KiB\n";
        assert_eq!(
            parse_amd_fdinfo(text),
            Some(AmdClient {
                pdev: "0000:43:00.0".into(),
                client_id: "39".into(),
                mem_used_kib: 15_203_992,
            })
        );
        assert!(parse_amd_fdinfo("drm-driver:\ti915\n").is_none());
    }

    #[test]
    fn module_flags_are_not_models() {
        // `-m` after an interpreter runs a module; these are not LLM servers.
        assert!(!looks_like_llm(
            "python3",
            "/usr/bin/python3 -m http.server 8470"
        ));
        assert!(!looks_like_llm(
            "gjs",
            "/usr/bin/gjs -m /usr/share/gnome-shell/org.gnome.Shell.Notifications"
        ));
        assert!(!looks_like_llm(
            "python3",
            "/usr/local/bin/python3 -m uvicorn open_webui.main:app --host 0.0.0.0"
        ));
        // A real server naming a weights file still matches.
        assert!(looks_like_llm(
            "llama-server",
            "/opt/bin/llama-server -m /models/Qwen3-4B-Q6_K.gguf -c 8192"
        ));
        assert!(looks_like_llm(
            "serve",
            "./serve --model-path mistralai/Mistral-7B"
        ));
    }

    #[test]
    fn merely_mentioning_gguf_is_not_a_server() {
        // A download, copy, or listing of a .gguf file is not an inference
        // server; only a --model/-m flag pointing at one counts.
        assert!(!looks_like_llm(
            "bash",
            "bash -c hf download unsloth/Qwen3.8-GGUF UD-Q3_K_XL/model-00002-of-00003.gguf --local-dir ."
        ));
        assert!(!looks_like_llm("cp", "cp /models/foo.gguf /mnt/backup/"));
    }

    #[test]
    fn keyword_mention_inside_a_wrapper_script_is_not_a_server() {
        // A monitoring/load-test loop that merely names the server in a
        // pgrep pattern or a path segment is not the server itself; only a
        // whole argv token (or its path basename) counts as a real mention.
        assert!(!looks_like_llm(
            "bash",
            r#"bash -c while pgrep -f "llama-server.*--port 8090" >/dev/null; do sleep 1; done"#
        ));
        assert!(!looks_like_llm(
            "bash",
            "bash /home/seth/epyc/vllm-b70/run_vllm.sh"
        ));
        // A real invocation still matches, whether bare or as a full path.
        assert!(looks_like_llm(
            "llama-server",
            "/opt/bin/llama-server -m /models/Qwen3-4B-Q6_K.gguf -c 8192"
        ));
        assert!(looks_like_llm("vllm", "/opt/venv/bin/vllm serve /model"));
    }

    #[test]
    fn extract_model_equals_form() {
        let p = parse_cmdline("python3", "python3 serve.py --model=gpt2");
        assert_eq!(p.name, "gpt2");
    }

    #[test]
    fn vllm_phantoms_are_filtered() {
        // The real server: the bare `vllm` main process, with or without
        // an explicit --port (8000 is vLLM's default).
        assert!(!is_vllm_phantom(
            "vllm",
            "/opt/venv/bin/vllm serve /model --port 8000"
        ));
        assert!(!is_vllm_phantom("vllm", "/opt/venv/bin/vllm serve /model"));
        // Named helpers inherit the parent command line -- including
        // --port -- so the process name, not the arguments, decides.
        assert!(is_vllm_phantom(
            "VLLM::EngineCore",
            "/opt/venv/bin/vllm serve /model --port 8000"
        ));
        assert!(is_vllm_phantom(
            "VLLM::Worker_TP0",
            "/opt/venv/bin/vllm serve /model --port 8000"
        ));
        // A docker client or wrapper that merely mentions vllm in its
        // arguments is not a server.
        assert!(is_vllm_phantom(
            "docker",
            "docker run --rm --name steve image vllm serve /model --port 8000"
        ));
        assert!(is_vllm_phantom(
            "bash",
            "bash /home/seth/epyc/vllm-b70/run_vllm.sh"
        ));
        // The comm is rarely the literal "vllm": python entrypoints and
        // renamed mains are still the real server.
        assert!(!is_vllm_phantom(
            "python3",
            "/usr/bin/python3 -m vllm.entrypoints.openai.api_server --model /m"
        ));
        assert!(!is_vllm_phantom(
            "pt_main_thread",
            "/opt/venv/bin/vllm serve /model"
        ));
        // nvidia-smi reports the executable path, not the comm.
        assert!(!is_vllm_phantom(
            "/opt/venv/bin/vllm",
            "/opt/venv/bin/vllm serve /model"
        ));
    }

    #[test]
    fn parse_vllm_serve_cmdline() {
        // argv[0] is a full path that must not be taken for the weights.
        let p = parse_cmdline(
            "vllm",
            "/opt/venv/bin/vllm serve /models/brain --port 8003 --max-model-len 40960 --served-model-name brain",
        );
        assert_eq!(p.engine, "vllm");
        assert_eq!(p.name, "brain");
        assert_eq!(p.path.as_deref(), Some(Path::new("/models/brain")));
        assert_eq!(p.port, Some(8003));
        assert_eq!(p.ctx_max, Some(40960));

        // Positional after unhandled flags: the flag's bare value must not
        // be taken for the weights either.
        let p = parse_cmdline(
            "vllm",
            "/opt/venv/bin/vllm serve --dtype float16 /models/qwen",
        );
        assert_eq!(p.path.as_deref(), Some(Path::new("/models/qwen")));
        assert_eq!(p.name, "qwen");

        // HuggingFace id as the positional.
        let p = parse_cmdline("vllm", "/opt/venv/bin/vllm serve Qwen/Qwen3-8B");
        assert_eq!(p.path.as_deref(), Some(Path::new("Qwen/Qwen3-8B")));
    }

    #[test]
    fn hf_config_ctx_reads_max_position_embeddings() {
        let dir = std::env::temp_dir().join("llm-visuals-test-hfcfg");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            "{\n  \"model_type\": \"qwen3\",\n  \"max_position_embeddings\": 40960,\n  \"vocab_size\": 151936\n}\n",
        )
        .unwrap();
        assert_eq!(hf_config_info(&dir).map(|g| g.ctx_train), Some(40960));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hf_config_reads_nested_text_config_and_experts() {
        let dir = std::env::temp_dir().join("llm-visuals-test-hfcfg-moe");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
              "model_type": "qwen3_moe",
              "text_config": {
                "model_type": "qwen3_moe",
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "max_position_embeddings": 40960,
                "hidden_size": 2048
              }
            }"#,
        )
        .unwrap();
        let g = hf_config_info(&dir).expect("config");
        assert_eq!(g.n_layers, 48);
        assert_eq!(g.n_heads, 32);
        assert_eq!(g.n_kv_heads, 4);
        assert_eq!(g.n_experts, 128);
        assert_eq!(g.n_experts_used, 8);
        assert_eq!(g.ctx_train, 40960);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_sglang_launch_cmdline() {
        let cmd = "/usr/bin/python3 -m sglang.launch_server --model-path /models/Qwen3.8-27B --port 30000 --context-length 40960 --speculative-algorithm EAGLE --kv-cache-dtype fp8";
        assert!(looks_like_llm("python3", cmd));
        let p = parse_cmdline("python3", cmd);
        assert_eq!(p.engine, "sglang");
        assert_eq!(p.port, Some(30000));
        assert_eq!(p.ctx_max, Some(40960));
        assert_eq!(p.spec_type.as_deref(), Some("EAGLE"));
        assert!(p
            .path
            .as_deref()
            .is_some_and(|p| p.ends_with("Qwen3.8-27B")));

        // Default port when --port is omitted.
        let p = parse_cmdline(
            "python3",
            "python3 -m sglang.launch_server --model-path /models/qwen",
        );
        assert_eq!(p.port, Some(30000));
        assert!(looks_like_llm(
            "python3",
            "python3 -m sglang.launch_server --host 0.0.0.0"
        ));
    }

    #[test]
    fn sglang_workers_are_folded() {
        assert!(is_sglang_worker("sglang::scheduler"));
        assert!(is_sglang_worker("sglang::detokenizer"));
        assert!(!is_sglang_worker("python3"));
        // /proc/<pid>/stat: comm in parens, then state, then ppid.
        let stat =
            "4321 (sglang::scheduler) S 1200 1200 1200 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 0 0 0";
        assert_eq!(ppid_from_stat(stat), Some(1200));
    }

    #[test]
    fn fib_trie_local_ips_parsed() {
        // A real /proc/<pid>/net/fib_trie (docker bridge netns).
        let txt = "Local:\n  +-- 0.0.0.0/0 3 0 5\n     |-- 0.0.0.0\n        /0 universe UNICAST\n     +-- 127.0.0.0/8 2 0 2\n        +-- 127.0.0.0/31 1 0 0\n           |-- 127.0.0.0\n              /8 host LOCAL\n           |-- 127.0.0.1\n              /32 host LOCAL\n        |-- 127.255.255.255\n           /32 link BROADCAST\n     +-- 172.17.0.0/16 2 0 2\n        +-- 172.17.0.0/30 2 0 2\n           |-- 172.17.0.0\n              /16 link UNICAST\n           |-- 172.17.0.3\n              /32 host LOCAL\n        |-- 172.17.255.255\n           /32 link BROADCAST\nBroadcast:\n  +-- 0.0.0.0/0 2 0 2\n     |-- 255.255.255.255\n        /32 link BROADCAST\n";
        let ips = fib_local_ips(txt);
        assert!(ips.iter().any(|i| i == "127.0.0.1"));
        assert!(ips.iter().any(|i| i == "172.17.0.3"));
        // Subnet headers carry a /prefix, and other sections' addresses
        // must not leak in.
        assert!(!ips.iter().any(|i| i.contains('/')));
        assert!(!ips.iter().any(|i| i == "255.255.255.255"));
    }
}
