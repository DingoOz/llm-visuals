# Roadmap

See `../README.md` for what the tool shows and `ARCHITECTURE.md` for how it
is put together.

## Done

- Auto-detects the running inference server and reads its GGUF header.
- Throughput, TTFT, tok/J and request log derived from `/slots` counters.
- GPU cards from `nvidia-smi` with peak-hold gauges, VRAM split, histories.
- Context bar, MTP acceptance panel (`/metrics`), layer tiles, expert heat map.
- Real router choices via a llama-server patch (`patches/`, `GET /experts`).
- Truecolor with automatic 256-colour fallback; `--demo` runs without hardware.
- `--log-db`: model, GPU and per-request history to SQLite, on by default.
- Settings screen (`s`): change options live or save them as launch defaults.
- Memory pipeline view (`b`): disk, RAM, PCIe, VRAM, prefill and decode VU
  meters with a bottleneck verdict; bytes per step from the GGUF tensor table.
- Native Windows: process detection and memory counters via `sysinfo`.
- SGLang: `/v1/loads` poller, `/server_info`, worker folding, safetensors
  `config.json` architecture fields (also used by vLLM).
- Decode speed vs context length (`c`), from the logged samples.
- Ollama: its llama-server runners polled through `/slots` and named by tag
  from the store's manifests; `/api/ps` for the unload countdown, the
  CPU/GPU split and the CPU-side bytes; daemon-only slots labelled
  `throughput n/a`.

## Ideas not yet done

- Per-position acceptance rates for deeper MTP drafts (server logs them but
  does not export them).
- Show the /experts 256-token histogram as a load-balance view per layer.
- Throughput for Ollama releases that predate the llama-server runner, and
  for a daemon whose runner is hidden (container): neither exports token
  counters, so it would take a proxy in front of the API.
- Real DRAM bandwidth (perf uncore counters) instead of the bytes × steps
  estimate; per-GPU memory bandwidth ceilings so the VRAM meter can show GB/s
  against the card's peak.
- Honour `-ot` / `--override-tensor` and `--fit` output to place tensors on
  CPU vs GPU exactly instead of by `--tensor-split` share.
