//! The safetensors counterpart of the GGUF tensor table.
//!
//! A Hugging Face model directory lists its tensors in the header of each
//! `*.safetensors` shard: 8 little-endian bytes of header length, then a
//! JSON object mapping tensor name to dtype, shape and byte range. Reading
//! only those headers costs kilobytes per shard and yields the same
//! `TensorSummary` the GGUF reader builds, so the memory-pipeline streams
//! work for safetensors servers too. The file bytes are exact; their
//! projection to bytes-per-token rests on the config's expert counts, so
//! the summary is flagged `estimated` and the UI says so.

use crate::gguf::TensorSummary;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// A safetensors header is a JSON table of tensor names; a real model's is
/// a few MB at worst. Anything larger is not a header.
const MAX_HEADER_BYTES: u64 = 64 << 20;

/// Weight bytes of a model directory. The index's shard list when the
/// directory has one, otherwise the top-level `*.safetensors` files.
/// `Err` when the directory holds no readable safetensors at all.
pub fn read_summary(dir: &Path) -> Result<TensorSummary, String> {
    let mut out = TensorSummary {
        estimated: true,
        ..Default::default()
    };
    let mut seen_any = false;
    for shard in shard_files(dir) {
        let Ok(tensors) = read_shard(&shard) else {
            continue;
        };
        seen_any = true;
        for (name, size) in tensors {
            accumulate(&mut out, &name, size);
        }
    }
    if !seen_any {
        return Err(format!(
            "no readable safetensors header under {}",
            dir.display()
        ));
    }
    Ok(out)
}

/// The shards to walk: the files named by `model.safetensors.index.json`
/// when present (it is authoritative and skips stray files), else every
/// top-level `*.safetensors`.
fn shard_files(dir: &Path) -> Vec<PathBuf> {
    if let Ok(txt) = std::fs::read_to_string(dir.join("model.safetensors.index.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
            if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
                // weight_map keys are tensor names; its values are the shards.
                let mut names: Vec<String> = map
                    .values()
                    .filter_map(|v| v.as_str())
                    .map(String::from)
                    .collect();
                names.sort();
                names.dedup();
                let files: Vec<PathBuf> = names
                    .into_iter()
                    .map(PathBuf::from)
                    .filter(|p| !p.is_absolute())
                    .map(|p| dir.join(p))
                    .filter(|p| p.is_file())
                    .collect();
                if !files.is_empty() {
                    return files;
                }
            }
        }
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
                .collect()
        })
        .unwrap_or_else(|_| Vec::new());
    out.sort();
    out
}

/// Read one shard's header: (name, byte size) per tensor. Payload bytes are
/// never touched — the read stops at the header JSON.
fn read_shard(path: &Path) -> Result<Vec<(String, u64)>, String> {
    let file_len = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len < 2 || header_len > MAX_HEADER_BYTES || header_len.saturating_add(8) > file_len {
        return Err(format!(
            "{}: implausible safetensors header",
            path.display()
        ));
    }
    let mut buf = vec![0u8; header_len as usize];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf);
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let obj = v
        .as_object()
        .ok_or_else(|| "safetensors header is not an object".to_string())?;
    let mut out = Vec::with_capacity(obj.len());
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let size = match meta.get("data_offsets").and_then(|d| d.as_array()) {
            Some(a) if a.len() == 2 => match (a[0].as_u64(), a[1].as_u64()) {
                (Some(b), Some(e)) if e > b => e - b,
                _ => continue,
            },
            _ => continue,
        };
        out.push((name.clone(), size));
    }
    Ok(out)
}

/// File name conventions are all there is to classify by; these are the HF
/// transformers names the GGUF table would spell `blk.N.*`, `token_embd`,
/// `per_layer_token_embd` and `ffn_*_exps`.
fn accumulate(out: &mut TensorSummary, name: &str, size: u64) {
    out.total_bytes += size;
    out.n_tensors += 1;
    if name.contains(".experts.") {
        out.expert_bytes += size;
    }
    if name.contains("embed_tokens")
        || name.contains("tok_embeddings")
        || name.ends_with(".wte.weight")
    {
        out.embd_bytes += size;
    }
    if name.contains(".ple.") || name.contains("per_layer_token_embd") {
        out.engram_bytes += size;
    }
    let parts: Vec<&str> = name.split('.').collect();
    for i in 1..parts.len() {
        if parts[i - 1] == "layers" {
            if let Ok(n) = parts[i].parse::<usize>() {
                if n < 4096 {
                    if out.block_bytes.len() <= n {
                        out.block_bytes.resize(n + 1, 0);
                    }
                    out.block_bytes[n] += size;
                }
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_shard(dir: &Path, name: &str, tensors: &[(&str, u64)]) -> PathBuf {
        let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
        let mut off = 0u64;
        for (t, size) in tensors {
            header.push_str(&format!(
                ",{t:?}:{{\"dtype\":\"F16\",\"shape\":[2,2],\"data_offsets\":[{off},{}]}}",
                off + size
            ));
            off += size;
        }
        header.push('}');
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&header.len().to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&vec![0u8; off as usize]).unwrap();
        path
    }

    #[test]
    fn summary_splits_experts_embd_and_blocks() {
        let dir = std::env::temp_dir().join(format!("lv-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_shard(
            &dir,
            "model-00001-of-00002.safetensors",
            &[
                ("model.embed_tokens.weight", 1000),
                ("model.layers.0.self_attn.q_proj.weight", 400),
                ("model.layers.0.mlp.experts.0.gate_proj.weight", 300),
                ("model.layers.0.mlp.experts.1.gate_proj.weight", 300),
                ("model.layers.1.self_attn.q_proj.weight", 400),
                ("model.layers.1.mlp.experts.0.gate_proj.weight", 300),
                ("model.layers.1.mlp.experts.1.gate_proj.weight", 300),
                ("model.ple.embedding.weight", 500),
                ("lm_head.weight", 100),
            ],
        );
        let s = read_summary(&dir).unwrap();
        assert!(s.estimated);
        assert_eq!(s.total_bytes, 1000 + 400 + 600 + 400 + 600 + 500 + 100);
        assert_eq!(s.expert_bytes, 1200);
        assert_eq!(s.embd_bytes, 1000);
        assert_eq!(s.engram_bytes, 500);
        assert_eq!(s.block_bytes, vec![1000, 1000]);
        assert_eq!(s.n_tensors, 9);
        // 2 experts, 1 used: half the expert bytes per token; the embedding
        // and the PLE table are row lookups and never streamed.
        assert_eq!(s.active_bytes_per_token(2, 1), 1500);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_list_wins_over_stray_files() {
        let dir = std::env::temp_dir().join(format!("lv-st-idx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_shard(&dir, "a.safetensors", &[("model.layers.0.a.weight", 10)]);
        write_shard(&dir, "b.safetensors", &[("model.layers.0.a.weight", 20)]);
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":10},"weight_map":{"model.layers.0.a.weight":"a.safetensors"}}"#,
        )
        .unwrap();
        let s = read_summary(&dir).unwrap();
        assert_eq!(s.total_bytes, 10);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_dir_is_an_error() {
        let dir = std::env::temp_dir().join(format!("lv-st-x-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(read_summary(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
