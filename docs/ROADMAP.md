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
- Expert balance view (`u`): the /experts 256-token histogram as per-layer
  effective experts, top-k share and a ranked-hits curve, with lopsided
  layers flagged.
- llama.cpp tensor overrides (`-ot`, `--cpu-moe`, `--n-cpu-moe`,
  `--n-cpu-ffn`) applied to the tensor table: exact CPU-side bytes for the
  memory pipeline and verdict, `C` tags on layer tiles.
- Per-GPU peak memory bandwidth from NVML (bus width × memory clock): the
  VRAM stage shows the weight stream in GB/s against it, and the verdict
  names VRAM bandwidth past 60 % of the peak.
- Ollama: its llama-server runners polled through `/slots` and named by tag
  from the store's manifests; `/api/ps` for the unload countdown, the
  CPU/GPU split and the CPU-side bytes; daemon-only slots labelled
  `throughput n/a`.
- GPU energy per session and per generated token, priced with `--energy-price`.
- Alerts (`--alerts`): VRAM growth past its settled level, GPU / memory hot, slow decode against the
  model's own baseline, server gone; banner, bell or desktop notification.
- Regression flag: decode speed now vs the same model's logged week, per
  context range and on the same GPU set.

## Ideas not yet done

- Per-position acceptance rates for deeper MTP drafts (server logs them but
  does not export them).
- Throughput for Ollama releases that predate the llama-server runner, and
  for a daemon whose runner is hidden (container): neither exports token
  counters, so it would take a proxy in front of the API.
- Real DRAM bandwidth (perf uncore counters) instead of the bytes × steps
  estimate.
- Peak memory bandwidth off NVML: AMD needs the `AMDGPU_INFO` ioctl's
  `vram_bit_width` (sysfs has only the clock).
- `--fit` output, to place the tensors it moves at load time (only the
  server's log says where).
