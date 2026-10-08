# llm-visuals

**A live terminal dashboard for the LLM running on your machine.**

It finds the inference servers you already have up (llama.cpp `llama-server`,
ollama, vLLM, SGLang, Strata, …), reads their counters and NVIDIA, AMD or Intel
GPU telemetry, and turns them into a truecolor picture of what the model is doing
right now: tokens per second, time to first token, GPU load and memory, context fill, speculative-decoding
acceptance, which layers are busy on which GPU, and, with a small server patch,
exactly which experts a mixture-of-experts model routed the last token through.

Every number on screen comes from the server, the driver, or arithmetic on
those. The one thing that can be a visual stand-in (expert identities without
the patch) is labelled as such on screen.

![dashboard](docs/dashboard.svg)

<sub>Live session, 140×44: Qwen3.6-35B-A3B (256 experts, MTP) on a GTX 1070 and
a Tesla P100 under llama.cpp, mid-request.</sub>

---

## Contents

- [Quick start](#quick-start)
- [Windows](#windows)
- [macOS](#macos)
- [Building and cross-compiling](#building-and-cross-compiling)
- [What you see](#what-you-see)
- [Several models at once](#several-models-at-once)
- [Keys](#keys)
- [Where every number comes from](#where-every-number-comes-from)
- [Server setup: metrics and real expert routing](#server-setup-metrics-and-real-expert-routing)
- [Command-line options](#command-line-options)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [Contributing and license](#contributing-and-license)

---

## Quick start

Requirements: a Rust toolchain (1.75+), `nvidia-smi` for NVIDIA GPU panels,
`xpu-smi` for Intel GPU panels or the Linux amdgpu driver for AMD GPU panels,
and a locally listening
`llama-server` for throughput panels. Nothing at all is needed for demo mode.

On Windows 11, see [Windows](#windows) for setup; on a Mac, see
[macOS](#macos). Prebuilt binaries for all three, on x86-64 and ARM64, are
attached to each [release](https://github.com/DingoOz/llm-visuals/releases) —
see [Building and cross-compiling](#building-and-cross-compiling) to make your
own.

```sh
git clone https://github.com/DingoOz/llm-visuals
cd llm-visuals
cargo run --release -- --demo      # synthetic server + two synthetic GPUs
cargo run --release                # attach to the LLM server that is running
```

Or install the binary:

```sh
cargo install --path .
llm-visuals
```

The dashboard auto-detects the server: it lists GPU compute processes, scans
`/proc` for anything that looks like an inference engine, parses the command
line for the model path, port, context size and tensor split, and reads the
GGUF header for the architecture. Press `r` to rescan at any time.

---

## Windows

Windows 11 support is new and has only been type-checked for Windows, not
run on it yet. The checklist below lists what to test; please report results
in an issue.

### What to install

1. **Visual Studio Build Tools** with the *Desktop development with C++*
   workload ([download](https://visualstudio.microsoft.com/visual-cpp-build-tools/)).
   Rust's default Windows toolchain links with it, and the bundled SQLite used by
   `--log-db` is compiled with its C compiler.
2. **Rust** via [rustup](https://rustup.rs) (`rustup-init.exe`), keeping the
   default `x86_64-pc-windows-msvc` toolchain.
3. **Git**: [Git for Windows](https://git-scm.com/download/win), or
   `winget install Git.Git`.
4. **NVIDIA driver** for GPU panels. Current drivers put `nvidia-smi.exe` in
   `C:\Windows\System32`; check with `nvidia-smi` in a new terminal.
5. **Windows Terminal** (preinstalled on Windows 11). The old console host
   renders the dashboard poorly.
6. Optional: **Python 3** from python.org with `pip install torch transformers`,
   only for the `--model <hf-id>` attention view. The dashboard runs `python`
   on Windows, so it must be on the `PATH`.

Then, in PowerShell:

```powershell
git clone https://github.com/DingoOz/llm-visuals
cd llm-visuals
cargo run --release -- --demo
cargo run --release
```

Windows Terminal is detected as truecolor (`WT_SESSION`); in any other
console, pass `--color truecolor` if colours look flat.

### What differs from Linux

- **Detection** lists processes through the OS (via the `sysinfo` crate)
  instead of `/proc`. Run the dashboard as the same user as the server.
  Windows hides the command line of a process started by another user or as
  administrator, so that server is matched by name only: it is assumed to be on
  its engine's default port (8080 for llama.cpp, 11434 for Ollama, 30000 for
  SGLang, 8000 for vLLM), and the model name, context size and GGUF details are
  missing. Starting the dashboard from an administrator terminal should also
  expose them.
- **GPU memory per process**: Windows drivers report it as `[N/A]`, so a
  server shows 0 MB in the model strip. Card-level VRAM is unaffected.
- **Memory pipeline (`b`)**: RAM totals and the server's resident memory come
  from the OS. There are no system-wide disk-read, page-cache or page-fault
  counters, so the DISK stage shows "no disk counters". The per-process read
  rate counts every read the process makes (files, pipes, sockets), so it runs
  higher than on Linux. PCIe traffic needs `nvidia-smi dmon`, which Windows
  drivers may not support; the meter switches itself off after three failures.
- **Servers in WSL2 or Docker** are reachable via their published localhost
  ports. Auto-detection probes local inference ports (e.g. 7000, 8000, 8080,
  11434, 30000), or you can point directly at the server with
  `--endpoint http://localhost:7000/v1` (or set `LLM_ENDPOINT`). Note that endpoints
  use plain HTTP over TCP (HTTPS / TLS is not supported). Specifying an explicit
  `--endpoint` (or `--model http://...`) attaches to that server directly and is
  exempt from `--pid` filtering.

### What to test

- [ ] `cargo build --release` completes (MSVC linker and SQLite C build found).
- [ ] `--demo` renders every view (`a` `p` `h` `m` `b` `v`) and `q` restores
      the terminal.
- [ ] Colours: truecolor is picked automatically in Windows Terminal.
- [ ] GPU panels show utilisation, VRAM, power and temperature from `nvidia-smi`.
- [ ] A native `llama-server.exe --metrics` is detected with the right model
      name, port and context size, and throughput updates during a request.
- [ ] Same server on a non-default `--port`.
- [ ] The Ollama app (`ollama.exe`, port 11434) is detected.
- [ ] A server started from an administrator terminal, with the dashboard in
      a normal one: detected by name on the default port?
- [ ] `r` rescans after a server is started or stopped; several servers at
      once show in the model strip.
- [ ] Memory pipeline: RAM and per-process read rate move while a model
      loads; whether the PCIe meter works or turns itself off cleanly.
- [ ] Model paths with spaces (e.g. under `C:\Users\Jane Doe\models`). The
      command line is split on whitespace, so the path and the GGUF details
      are likely to be missing.
- [ ] With no flags, rows are written to `%LOCALAPPDATA%\llm-visuals\llm.db`;
      `--log-db llm.db` writes there instead; a path that cannot be created
      (e.g. on a missing drive) fails with a readable error.
- [ ] `s`, change a value, `w`: `%APPDATA%\llm-visuals\settings.json` is
      written and the value is used at the next launch.
- [ ] `--model <hf-id>` starts the Python bridge.
- [ ] Resizing the window and very small window sizes.

---

## macOS

macOS support is new and has only been type-checked, not run on a Mac yet.
Both Apple silicon and Intel are built and released; please report results in
an issue.

Requirements are the same as elsewhere, minus the GPU panels: `nvidia-smi`
does not exist on macOS, so the GPU row and the PCIe stage of the memory
pipeline stay empty. Everything driven by the server's own metrics —
throughput, context, MTP acceptance, layers and experts — works as it does on
Linux.

### What differs from Linux

- **Detection** lists processes through the OS (via the `sysinfo` crate)
  instead of `/proc`, exactly as on Windows. Run the dashboard as the same
  user as the server, or the server's command line — and with it the model
  path, port and context size — is hidden and it is matched by name alone.
- **No GPU panels**: there is no `nvidia-smi`, so utilisation, VRAM, power and
  temperature are unavailable, and Metal/unified memory is not read.
- **Memory pipeline (`b`)**: RAM totals and the server's resident memory come
  from the OS; the DISK and PCIe stages have no counters to read.
- **Settings** go to `~/Library/Application Support/llm-visuals/settings.json`
  and the log database to the same directory.

---

## Building and cross-compiling

`cargo build --release` on the machine you want to run on is always the
simplest route, and needs only a Rust toolchain and a C compiler (the SQLite
used by `--log-db` is bundled as C source and built from scratch).

Six targets are built and released:

| | x86-64 | ARM64 |
|---|---|---|
| **Linux** | `x86_64-unknown-linux-gnu` | `aarch64-unknown-linux-gnu` |
| **macOS** | `x86_64-apple-darwin` | `aarch64-apple-darwin` |
| **Windows** | `x86_64-pc-windows-msvc` | `aarch64-pc-windows-msvc` |

`.github/workflows/release.yml` builds all six on GitHub Actions — each on a
runner of its own architecture, except the Intel Mac binary which
cross-compiles from the ARM64 macOS runner — and attaches them to the release
for a pushed `v*` tag. `.github/workflows/ci.yml` compiles and tests on all
three operating systems on every pull request, and additionally cross-compiles
the ARM and Windows targets that no runner there covers.

### Cross-compiling locally

`scripts/cross-build.sh` builds the same set from one machine and writes
archives to `dist/`:

```sh
./scripts/cross-build.sh                 # every target this host can link
./scripts/cross-build.sh windows-arm64   # or one at a time
```

Anything other than the host target needs [zig](https://ziglang.org/download/)
and `cargo-zigbuild`, which supply the cross C compiler and linker that the
bundled SQLite needs:

```sh
cargo install cargo-zigbuild
pipx install ziglang          # or: brew install zig, or a tarball on the path
```

Two things to know about the result:

- **Windows** binaries are built for the `*-pc-windows-gnullvm` targets rather
  than `*-pc-windows-msvc`, because those are the ones zig can link. They are
  ordinary `.exe` files that need no toolchain installed to run; the released
  binaries are the msvc ones, built natively.
- **macOS** binaries cannot be cross-compiled without Apple's SDK, which is
  not redistributable: `sysinfo` links against the `CoreFoundation` and
  `IOKit` frameworks, which zig does not carry. The script skips those two
  targets unless `SDKROOT` points at a `MacOSX.sdk` you already have. Build
  them on a Mac, or let the release workflow do it.

Linux binaries are linked against glibc 2.28 by default so they run on
distributions older than the build host; set `GLIBC=` to link against the
host's own instead.

---

## What you see

### Throughput

![perf view](docs/perf.svg)

<sub>Performance zoom (`p`): tall history for decode and prefill rates, per-GPU
utilisation and power, and a longer request log.</sub>

Decode tokens/sec in large numerals, prefill tokens/sec, time to first token,
tokens per joule across all GPUs, session totals, and history sparklines.
Rates are computed over a one-second sliding window, which matters: speculative
decoders land tokens in bursts, and a per-poll rate flickers. On a test request
the server reported 24.3 tok/s and 412 ms prompt time; the dashboard showed
23.8 tok/s and a 0.38 s time to first token.

### GPUs

One card per device: utilisation gauge with a VU-style peak-hold notch, power
gauge, temperature coloured by heat, SM clock, fan, PCIe link, a VRAM bar split
into weights / KV in use / KV reserved / free, and utilisation and power
history sparklines.

Below the cards, a system RAM section shows memory in use / page cache / free
against the machine's total, the focused model's resident size, and a history
of RAM in use. It is on both the dashboard and the perf view, and is the first
thing the panel drops on a short terminal.

### Vision encoder

When the model has a vision encoder loaded, the header says where it runs:
`vision CPU`, or the card in the GPU panel's numbering followed by the
engine's own name for it, e.g. `vision G1 RTX 3070 (CUDA0)`. llama.cpp puts
the projector (`--mmproj`) on a device of its own, independent of
`--device` and `--tensor-split`: the first CUDA device unless
`--no-mmproj-offload` or `--mmproj-device` (`MTMD_BACKEND_DEVICE`) says
otherwise. CUDA numbers cards fastest-first by default, not in nvidia-smi's
PCI order, so `CUDA0` is often not `G0`; the dashboard asks the CUDA driver
which card each ordinal is, under the server's own `CUDA_VISIBLE_DEVICES` and
`CUDA_DEVICE_ORDER`. vLLM and SGLang shard the vision tower over the model's
own GPUs (`vision G0+G1`).

### Context and MTP

The context bar shows cached, prompt and generated tokens against the window
with a pulsing head while the model works, plus cache-hit rate, KV cache types,
flash-attention and slot occupancy. Beside it the MTP panel shows speculative
decoding health: windowed acceptance rate, mean accepted tokens per
verification step, verification steps per second, an acceptance history and
session totals.

### Layers and experts

![experts view](docs/experts.svg)

<sub>Expert zoom (`m`): one block per expert per layer, two experts per block at
this width, showing the router's real choices from `GET /experts`. Colour is
time since the expert was routed: white now, cooling to dark over about 0.6 s.</sub>

The layer panel draws one tile per transformer layer, tagged with the GPU it
lives on, coloured by that GPU's load with fast attack and slow release, each
carrying its own small history. The expert panel, for MoE models, draws one
block per expert per layer. With the server patch described below, the blocks
show the router's real choices and the title reads `live · N/256 active` (N is
the mean number of distinct experts each layer touched over its last 256
tokens). Without the patch the title reads `simulated`: the timing is real
(each new token re-routes every layer) but the identities are a stand-in.

If the model carries an **engram** module (a hashed n-gram memory; llama.cpp's
`<arch>.ple.*` keys, as in Qwen3.8-Flash-Next), the bottom row of the expert
panel describes it: n-gram order, the layer it feeds, hash heads, table rows ×
width, and the table's size on disk. It is one row lookup per token rather
than a routed matmul, so it is not drawn as a grid and its bytes are left out
of the per-token bandwidth estimate. The model strip shows `engram N-gram`.

### Memory pipeline

Press `b` for the bandwidth view: six VU-style channel strips, one per hop
the weights take on the way to a token, with an arrow animating between the
stages that are moving data and a verdict line naming the hop that is
holding the model back.

![bandwidth view](docs/bandwidth.svg)

<sub>Bandwidth view (`b`): a dense 27B Q6 that leaves ~4.7 GB on the CPU.
GPUs under 40 % busy, memory controllers well under a quarter, and the
verdict names RAM as the bound at 2.5 tok/s.</sub>

| Stage | Meter | Source |
|---|---|---|
| DISK | MB/s read from every whole block device | `/proc/diskstats`, plus the server's own reads and major page faults from `/proc/<pid>/io` and `/proc/<pid>/stat` |
| RAM | GB/s of weights the CPU streams out of system RAM, *estimate* | CPU-side bytes × active fraction × steps/s; CPU-side bytes = tensor-table size minus what the cards hold. Tensor table = the GGUF tensor table, or the `*.safetensors` shard headers (header bytes only, labelled `est`) for safetensors servers. A safetensors server has no partial offload, so with a GPU present none of its weights count as CPU-side |
| PCIe | host→device MB/s per GPU, scaled to the link (gen × lanes) | `nvidia-smi dmon -s t` (NVIDIA; unavailable for AMD). On a unified part the stage is labelled `C2C` and reads `n/a`: the traffic is NVLink-C2C and the driver exposes no counter for it |
| VRAM | memory-controller busy % per GPU, plus the estimated GB/s of weights streamed | NVIDIA `utilization.memory` or AMD `mem_busy_percent`; bytes per step from the tensor table. On a unified part the stage is labelled `UNIFIED` and shows pool fill (server-reported occupancy over system RAM) instead — there is no memory controller to watch |
| PREFILL | prompt tokens/s | `/slots` |
| DECODE | generated tokens/s | `/slots` |

"Bytes per step" is read from the tensor table: every tensor except the
embedding lookup, with `ffn_*_exps` tensors scaled by
`expert_used_count / expert_count`, so a 35B-A3B MoE reads ~2.7 GB per token
while a dense 27B Q6 reads ~24 GB. A GGUF server reports the table itself; a
safetensors server (SGLang, vLLM) gets the same summary from the headers of
its `*.safetensors` shards — file sizes are exact, the per-token projection
rests on the config's expert counts, so every number derived from it is
labelled `est`. The directory is found through the HF hub cache when the
server names a repo (`RadixArk/Model-NVFP4`) or an alias of a cached repo
instead of a host path: the snapshot `refs/main` points at, and for an alias
only when it is the start of exactly one cached model's name — two quants
under one alias are left unresolved rather than guessed. A step is one
verification pass under MTP / speculative decoding (from `/metrics`), one
token otherwise, and one micro-batch (`-ub`, default 512) during prefill. The verdict is a rule
chain: disk activity beats everything (weights are paging), then a PCIe link
past a third of its cap, then a memory controller past 75 %, then CPU-side
layers with an idle GPU, then a busy GPU (compute bound); otherwise no hop is
saturated and the gap is latency between tokens.

On a 27B Q6 model that does not quite fit two cards (~4.7 GB left on the
CPU), the view shows the GPUs under 40 % busy, memory controllers at 24 %,
and flags RAM as the bound at under 3 tok/s, which is what a CPU-offloaded
layer set feels like.

### Requests

One row per server task: prompt, cached and generated tokens, average prefill
and decode rates, time to first token and duration. The live request pulses.

---

## Several models at once

Every inference server on the machine is monitored, not just the first one
found. Each gets its own poller, its own rate windows and request log, and its
own smoothing, so one model prefilling does not disturb another's numbers.

**Shown together.** A strip under the header carries one line per model —
phase, decode rate, recent history, context fill and VRAM — so you can see at
a glance which one is working. The GPU panel tags each card with the models
resident on it (`⟨1,3⟩`), since they share the box.

**Shown side by side.** `v` opens the comparison view: one card per model with
its rate, prefill, TTFT, context, acceptance, shape, VRAM and history, up to
four across, wrapping onto more rows beyond that.

**Shown one at a time.** The panels that only make sense for a single model —
layer tiles, the expert map, the memory pipeline, the request log — follow the
focused model. `Tab` / `Shift-Tab` move the focus, `1`–`9` jump straight to
one, and the header says which of how many you are looking at.

The strip appears only when there is more than one model, so a single-server
setup looks exactly as it did before. `--max-models N` caps how many are
watched and `--pid A,B` restricts it to named processes.

---

## Keys

| Key | Action |
|-----|--------|
| `a` | full dashboard |
| `p` | performance zoom: tall sparklines, longer request log |
| `h` | layer tiles zoom |
| `m` | expert map zoom, finer blocks |
| `b` | memory pipeline: disk → RAM → PCIe → VRAM → prefill → decode VU meters and the bottleneck verdict |
| `v` | compare every model side by side |
| `Tab` / `Shift-Tab` | focus the next / previous model |
| `1`–`9` | focus that model directly |
| `t` | cycle theme (panel frames, backgrounds and heat maps): defrag, neon, fire, ocean, monochrome, braille (btop-style braille graphs) |
| `r` | rescan for running servers |
| `s` | settings screen (shows the version): change launch options, apply them now or save them as the default |
| `l` | log viewer: per-model totals and the latest requests from the SQLite log (`r` refreshes) |
| `c` | decode speed vs context length from the SQLite log, one model at a time (`←` `→` switch model, `r` refreshes) |
| `q` / `Esc` | quit |

Layouts adapt: the model strip is the first thing shed on a short terminal,
then the request log, then the context and MTP row, and on narrow terminals
the GPU line sheds PCIe, fan, clock and temperature before the gauge shrinks,
while the key row shortens its own labels. Truecolor is auto-detected with a
256-colour fallback.

---

## Where every number comes from

| Metric | Source |
|---|---|
| decode tok/s | llama.cpp: delta of `n_decoded` from `GET /slots`. When that field is absent, delta of `llamacpp:tokens_predicted_total` from `GET /metrics`, anchored at the start of the request. vLLM: `/metrics` generation counter. SGLang: `decode_moments[5]` from `GET /v1/loads`. Strata: `live.generated` from `GET /metrics`. 1 s sliding window. Polls go to the server's `--host` (loopback when it bound `0.0.0.0`) |
| prefill tok/s | llama.cpp: `n_prompt_tokens_processed`. vLLM: prompt-token counter. SGLang: `total_prefill_uncached_tokens`, or `sglang:realtime_tokens_total{mode="prefill_compute"}` with `--enable-metrics`. Strata: engine-measured `live.prefill_tok_s_mean`, then uncached tokens / `prompt_ms` at completion, from JSON `GET /metrics` |
| time to first token | slot turning busy → first decoded token, quantised to the poll interval. Strata: the finished request's `prompt_ms`, the server's time reading the new prompt tokens (time spent queued is not in it) |
| tok/J | decode tok/s ÷ summed GPU power draw |
| cache hit | llama.cpp: `n_prompt_tokens_cache / n_prompt_tokens`. SGLang without `--enable-metrics` is unknown (shown as "—"). Strata: the finished request's `reused`, unknown ("—") while it runs |
| request log | one record per `id_task`; averages from accumulated deltas |
| util, VRAM, power, °C, clocks, fan, PCIe link | NVIDIA in-process NVML (`nvidia-smi --query-gpu=…` fallback), Intel `xpu-smi --query-gpu=…`, or Linux amdgpu sysfs and hwmon, every poll. On Intel, fan speed is the `xe` driver's hwmon tachometer (RPM); utilization that samples ~0 while clocks are boosted is reconstructed from the clock ratio and marked `~`; PCIe link and an unsupported temperature read blank. Unified-memory parts (NVIDIA GB10 / DGX Spark) report no device memory, fan or memory-busy counter at all: NVML stays the backend for the fields that *do* answer, and the memory bar relabels itself (see `VRAM weights vs KV` and Troubleshooting) |
| system RAM | Linux `/proc/meminfo` (`MemTotal − MemAvailable` in use, `Cached` as page cache) and the server's `VmRSS`; elsewhere the OS memory totals, with no page cache split |
| VRAM weights vs KV | llama.cpp: **estimate** from GGUF file size × `--tensor-split`. SGLang: `memory.weight_gb` and `memory.kv_cache_gb` from `/v1/loads`. vLLM and other safetensors servers: **estimate** from the summed size of the served directory's weight shards. On a part with no device memory (GB10 / DGX Spark) the bar is labelled `UNIFIED`: used = the server's weight + KV + CUDA-graph GiB from `/v1/loads` summed over the models that report it, denominator = system RAM — the pool the accelerator shares. With no server-reported occupancy the row degrades to zeros rather than a stand-in |
| layers, heads, experts, MTP layers, engram, quant | GGUF header, or HuggingFace `config.json` (`num_hidden_layers`, `num_attention_heads`, `num_experts` / `num_local_experts`, `num_experts_per_tok`) for safetensors dirs |
| vision encoder | llama.cpp: `modalities.vision` from `GET /props`, else `--mmproj` / `-hf` on the command line or `LLAMA_ARG_MMPROJ` in `/proc/<pid>/environ`. Placement: CPU for `--no-mmproj-offload` or a process with no GPU runtime in `/proc/<pid>/maps`; otherwise `--mmproj-device` or the first GPU device, mapped to a card by the CUDA driver's `cuDeviceGetPCIBusId` (run in a child process with the server's CUDA variables) matched against nvidia-smi's `pci.bus_id`, or by the one card the process occupies. vLLM / SGLang: `vision_config` in `config.json`, on the model's GPUs (none with `--language-model-only`) |
| layer → GPU | `--tensor-split` proportions |
| layer activity | utilisation of the GPU the layer lives on, smoothed |
| expert blocks | real top-k routing from `GET /experts` (patched server), else a deterministic stand-in keyed by layer and token step |
| MTP / speculative depth (tokens drafted per step) | llama.cpp: `--spec-draft-n-max` (or `--draft-max`) on the command line, else the model's MTP layer count. vLLM: number of per-position acceptance counters. SGLang: `speculative_num_draft_tokens` |
| MTP acceptance, tok/step, steps/s | deltas of `spec_decode_num_draft_tokens_total`, `…accepted_tokens_total`, `…drafts_total` from `GET /metrics`, 1.5 s window |
| disk MB/s, faults/s | deltas of sectors read in `/proc/diskstats` (whole disks), `read_bytes` in `/proc/<pid>/io`, `majflt` in `/proc/<pid>/stat` |
| resident weights | `RssFile` in `/proc/<pid>/status` |
| PCIe MB/s | in-process NVML `nvmlDeviceGetPcieThroughput` (`nvidia-smi dmon -s t -c 1` fallback) rx/tx per GPU; samples above the link cap are dropped (dmon emits the odd garbage row) |
| VRAM busy % | NVIDIA `utilization.memory` or AMD `mem_busy_percent` (memory-controller busy time) |
| bytes per step, RAM / VRAM GB/s | **estimate**: GGUF tensor table (sizes from offset gaps, summed over every shard of a split file), expert tensors × used/total, embedding and engram tables excluded, split CPU vs GPU by what the cards hold, × steps/s |

Per-request `timings` only appear inside completion responses, which the
dashboard never sees, so everything is reconstructed from polled counters.

---

## Server setup: metrics and real expert routing

Throughput and context work with any llama-server. Two panels need more:

**MTP / speculative acceptance** needs the Prometheus endpoint:

```sh
llama-server ... --metrics          # or env LLAMA_ARG_ENDPOINT_METRICS=1
```

**Real expert routing** needs a small patch to llama.cpp, included in
[`patches/`](patches/README.md). Stock llama.cpp computes the router's top-k
choices in every MoE layer but never exports them; the patch adds an
`--expert-stats` flag and a `GET /experts` endpoint.

```sh
cd /path/to/llama.cpp
git apply /path/to/llm-visuals/patches/llama-server-expert-stats.patch
cmake --build build --target llama-server -j
llama-server ... --metrics --expert-stats   # or env LLAMA_ARG_EXPERT_STATS=1
```

For a systemd unit, a drop-in with two `Environment=` lines is enough; see
[`patches/README.md`](patches/README.md). The dashboard probes both endpoints
at start and after `r`, and stops asking after three failures, so unpatched
servers cost nothing.

### API keys and LM Studio

A llama-server started with `--api-key` or `--api-key-file` answers `/slots`,
`/metrics` and `/props` with 401. The dashboard reads the key from that
server's command line and sends it as a bearer token, so no setup is needed.
If the key is not on the command line, `--api-key-file` supplies one for
every server.

LM Studio starts every model it loads as such a llama-server, with its own
port and key, so press `r` after it loads a different model. It has no switch
for `--metrics`, but the llama-server it starts inherits its environment:
launch LM Studio with `LLAMA_ARG_ENDPOINT_METRICS=1` set to get the MTP panel.

### SGLang

SGLang is detected from `python -m sglang.launch_server` (and the `sglang`
keyword). Throughput comes from `GET /v1/loads?include=all`, which needs no
flags. Context length and the speculative algorithm come from
`GET /server_info`. The scheduler and detokenizer worker processes are folded
into the launcher so they do not appear as extra models.

Each poll is a line in SGLang's access log, so the SGLang poller never runs
faster than 400 ms. To silence those lines:

```sh
python -m sglang.launch_server ... \
  --uvicorn-access-log-exclude-prefixes /v1/loads /metrics /server_info
```

Optional `--enable-metrics` adds exact prefill and prefix-cache counts via
`sglang:realtime_tokens_total{mode="prefill_compute"|"prefill_cache"}`.
Without it, short prompts after idle can show no prefill rate, and cache hit
is shown as "—" rather than 0. Speculative acceptance uses SGLang's own
arithmetic: accepted = generated − verify steps, drafted = steps × (draft
tokens − 1). A separate draft model titles the panel SPECULATIVE rather than
MTP.

### Strata

[Strata](https://github.com/Niko1221/Strata) is detected from
`serve/server.py --engine strata`, on its `--port` (8095 when absent). The
model name, the first GGUF shard (`--native`) and `--max-context` are read
from the `--config` JSON, so the header shows the model's layers and experts.
The native `strata --serve` child holds the GPU memory and is folded into the
server process.

Everything live comes from Strata's JSON `GET /metrics`, polled no faster
than every 400 ms. From Strata 0.1.40.2 that endpoint answers a client that
accepts `text/plain` in Prometheus text under vLLM's metric names, so the
poll asks for `application/json` alone; a server reached by `--endpoint` is
recognised by the `strata:` samples in that text and then read as JSON
(without that it would be taken for vLLM). Prefill progress and generated tokens come from `live`;
the prefix reuse and the server's own prefill and decode times are only
known once a request ends, so while one runs cache hit shows "—". A
live prefill rate uses the engine's `live.prefill_tok_s_mean`, not progress
divided by a poll interval (progress includes reused prefix positions).
On engines without that field, the rate stays zero until completion; the
request table and completion sample use only new tokens divided by
`prompt_ms`, including partial cancelled reads. The completion sample moves
the meter only for a prefill it never showed live, so a request draws one
prefill burst, not a second one after its decode. Back-to-back requests retain
their own timings even when no idle poll separates them. A
batching engine (`live.parallel`) is followed by its newest request. The MTP
depth is `engine.mtp_max`. From Strata 0.1.35 the acceptance gauge comes
from the `totals.drafts_offered` / `drafts_accepted` counters; they move only
when a request ends, so the gauge shows each finished request's acceptance
(spread over its server-measured decode time) and holds it until the next. The VRAM weights/KV split is the usual
file-size estimate, which does not fit Strata's RAM-resident experts.

---

## Command-line options

```
--demo               synthetic servers and GPUs; exercises every panel
--demo-models N      how many synthetic servers --demo runs (default 2)
--endpoint <url>     inference server endpoint URL (e.g. http://localhost:7000/v1)
--model <id|url|auto>`auto` (default) observes running servers or local endpoints;
                     an HTTP URL attaches to that inference endpoint;
                     an HF id streams real attention via the Python bridge
--max-models N       most models to watch at once (default 8)
--pid A,B            only watch these PIDs (default: every model found)
--gpu 0,1            GPU indices to show (default: all)
--no-nvml            use nvidia-smi subprocesses instead of native NVML telemetry
--poll-ms 200        sampling interval for the server and GPU telemetry
--api-key-file PATH  bearer token file for inference-server HTTP requests
--color auto|truecolor|256
--theme defrag|neon|fire|ocean|monochrome|braille
--max-layers N, --max-heads N     caps for the attention view
--log-db auto|off|FILE  SQLite log of samples and requests (default: auto)
--log-every 1.0      seconds between --log-db sample rows
--log-db-max-mb 1024 size cap for --log-db; oldest rows are dropped (0 = none)
```

`llm-visuals --help` lists everything.

When the inference server requires authentication, point `--api-key-file` at
a file containing only the bearer token. The token is read at launch, is never
stored in saved settings or the metrics database, and is attached to all HTTP
probes made to the detected inference server. Keep the file readable only by
the account running the dashboard.

### Settings screen and saved defaults

Press `s` to change the theme, colour depth, poll interval, model limits,
GPUs and logging while the dashboard runs. `a` applies the values to this
session; `w` also saves them as the launch default, in
`~/.config/llm-visuals/settings.json` (`%APPDATA%\llm-visuals` on Windows,
`~/Library/Application Support/llm-visuals` on macOS). The file maps flag
names to values, e.g. `{"theme": "neon", "poll-ms": "500"}`. Saved values are
read as if typed before your own flags, so a flag on the command line still
wins. Only values that differ from the built-in default are kept. The GPU
selection takes effect at the next launch; everything else applies at once.
If the file holds a bad value, the dashboard starts without it and says why
in the status line.

### SQLite log

Logging is on by default. `--log-db auto` writes to
`~/.local/share/llm-visuals/llm.db` (`%LOCALAPPDATA%\llm-visuals` on
Windows, `~/Library/Application Support/llm-visuals` on macOS); pass a path
to use another file, or `--log-db off` to turn it off. `--demo` does not log
unless given an explicit path, so synthetic numbers stay out of the real
history. If the default location cannot be opened, the dashboard keeps
running without logging; an explicit path that fails is an error.

Three tables are written (timestamps are Unix seconds):
`model_samples` (decode/prefill tok/s, context fill, session totals per model),
`gpu_samples` (utilisation, VRAM, power, temperature per card) and `requests`
(one row per finished request: tokens, TTFT, duration, average rates). The file
uses WAL, so it can be queried while the dashboard runs. Once the data passes
`--log-db-max-mb` (1 GB by default) the oldest tenth of each table is deleted;
SQLite reuses the freed pages, so the file stops growing at about that size.

Press `l` for a read-only view of the file: row counts, the time span
covered, per-model request totals with average decode rate and TTFT, and the
newest requests. With logging off it shows the default file from earlier
sessions. Press `c` to see how decode speed falls as the context fills: the mean
decode tok/s of every logged sample, grouped into about a dozen context-length
ranges per model, with the change from the shortest to the longest range. Only
samples taken while the model was decoding count; ones that overlap a prefill
are left out, since their one-second window is partly empty. Ranges with fewer
than five samples are dimmed. It opens on the focused model; `←` / `→` switch
to the others in the file.

For anything else, query it directly:

```sh
sqlite3 ~/.local/share/llm-visuals/llm.db "SELECT model, AVG(avg_decode_tps) FROM requests GROUP BY model"
```

---

## Troubleshooting

**"no inference server detected".** The server must be a local process; the
detector looks for `llama-server`, `ollama`, `vllm`, `sglang`, `exllama`,
`text-generation` and similar names, or a `--model` / `-m` flag whose argument
names a weights file, an existing directory or a HuggingFace id. (A bare
`-m` after an interpreter is a module, not a model, so `python3 -m uvicorn`
is not mistaken for a server.) Start it, then press `r`.

**A model you are running is missing from the strip.** An engine daemon with
nothing loaded is hidden whenever a server that is actually serving a model is
present, and `--max-models` caps the list at 8 by default. Raise the cap, or
name the process with `--pid`.

**GPU panel says "Driver/library version mismatch".** `nvidia-smi` itself is
failing: the NVIDIA userspace was upgraded under a running kernel module.
Reload the modules or reboot. The panel shows whatever `nvidia-smi` prints so
the cause is visible.

**Fewer GPU fields on a GB10 / DGX Spark (unified-memory) machine.** Nothing is
broken: the driver exposes no device memory, no fan tachometer and no
memory-controller counter there. The GPU panel stays on in-process NVML for
utilisation, power, temperature and clocks; the VRAM bar becomes `UNIFIED` with
system RAM as the denominator and the SGLang server's own
`memory.weight_gb + kv_cache_gb + graph_gb` from `/v1/loads` as the fill (other
engines do not report occupancy, so the bar shows zeros). Fan and `VRAM busy %`
read as absent, not as errors. The two bars show the same pool at different
units and different scopes: `UNIFIED` is GiB like every VRAM bar and counts
only what the servers report as model occupancy; the `RAM` row is decimal GB
(`/ 1e9`) and counts the whole system (`MemTotal − MemAvailable`). The same
119.6 GiB pool is 128.5 G and 119 GiB to btop. The `b` memory pipeline says the
same thing in its own labels: the PCIe hop becomes `C2C` reading `n/a` (no
bandwidth counter exists on a unified part) and the VRAM hop becomes `UNIFIED`
showing pool fill. Neither absent sensor may decide the bottleneck verdict —
on a unified part a saturated-looking bus is not reportable, so the verdict
says "GPU compute bound … no memory-controller or C2C counter exists here".
The RAM / VRAM stream estimates still work there: SGLang names a HF repo, the
weights are found in `~/.cache/huggingface/hub`, and the shard headers give the
same tensor table a GGUF file would (labelled `est`).

**AMD GPU panel is unavailable.** AMD telemetry requires Linux with the
`amdgpu` driver and readable DRM sysfs/hwmon files under `/sys/class/drm`.
No ROCm installation or privileged access is required. On other operating
systems the dashboard continues to use the existing NVIDIA collector.

**MTP panel says "start llama-server with --metrics".** Exactly that; see
above.

**Expert panel says "simulated".** The server does not serve `/experts`. Apply
the patch, rebuild, restart with `--expert-stats`, then press `r`.

**Expert panel says "dense model".** The GGUF has no `expert_count`; there is
nothing to route.

**Colours look flat.** Your terminal did not advertise truecolor. Run with
`--color truecolor`, or export `COLORTERM=truecolor`.

**PCIe strip says "PCIe throughput unavailable".** Live PCIe throughput is
currently NVIDIA-only. The driver may not report it for a particular NVIDIA
card, and AMD cards still show their link generation and width but not live
PCIe traffic. The other stages continue to work. Press `r` to retry.

**RAM strip says "no tensor table".** The model path on the server's command
line could not be opened as a GGUF (ollama blobs, remote paths, or a
non-GGUF engine). Disk, PCIe and VRAM meters still work; the byte estimates
need the file.

**Throughput reads zero while the model is clearly working.** Another client
may be using a different slot; the dashboard follows the busy slot when there
is one. Check `GET /slots` on the server.

**Context, cache-hit and decoded counters sit at zero on a multi-slot server.**
Known limitation. `llama-server` started with more than one slot (`-np`, or the
default on recent builds) only fills in `n_prompt_tokens`,
`n_prompt_tokens_cache`, `n_prompt_tokens_processed` and `next_token` on slots
that have actually served a request; the untouched ones are stubs carrying
`id`, `n_ctx`, `speculative` and `is_processing` and nothing else. The
dashboard follows the processing slot while a request is in flight, but as soon
as the server goes idle it falls back to slot 0 — which is a stub unless slot 0
happened to be the one that ran. The numbers then read zero between requests
even though the server is healthy and the last request was served fine. Check
with:

```bash
curl -s localhost:8080/slots | python3 -c \
  'import json,sys; [print(s["id"], sorted(k for k in s if k != "params")) for s in json.load(sys.stdin)]'
```

If only one slot in that listing carries the token keys and it is not slot 0,
this is what you are hitting. Running the server with `-np 1` avoids it.

---

## How it works

`docs/ARCHITECTURE.md` has the module map and data contracts. In short:
one poller per model reads its `/slots`, `/metrics` and `/experts` while
shared collectors read NVIDIA, AMD or Intel GPU telemetry and the host's `/proc`
counters every 200 ms into channels; samples are tagged with the model's PID, and the frame
loop routes each into that model's `PerfTracker` (sliding-window rates,
request lifecycle, peak hold) and `FadeState` (attack/release smoothing,
expert heat), then renders with ratatui at about 30 fps. Tests cover every
parser against captured real payloads.

```
src/
├── main.rs          event loop, per-model slots, pollers, key handling
├── render.rs        panels, gauges, sparklines, big digits
├── perf.rs          rates, TTFT, request records, MTP stats, VU meters
├── bandwidth.rs     weight layout and the bottleneck verdict
├── host.rs          /proc disk, faults, RSS; in-process NVML PCIe (dmon fallback)
├── fade.rs          smoothing and expert heat
├── observe.rs       /slots, /metrics, /experts parsers
├── gpu.rs           NVIDIA/AMD/Intel collectors, demo GPUs
├── nvml.rs          in-process NVML driver bindings & PCIe throughput
├── model_detect.rs  finds the servers, parses their command lines
├── gguf.rs          GGUF header reader, layer → GPU mapping
├── demo.rs          synthetic servers for --demo
├── settings.rs      settings screen, saved defaults, per-user paths
├── dblog.rs         SQLite log (--log-db)
├── colors.rs        palette, gradients, truecolor/256 gating
└── llm/             optional HF transformers attention bridge
patches/             llama.cpp patch for GET /experts
docs/                architecture, roadmap, screenshots
```

---

## Contributing and license

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md)
for the principles (real numbers only, degrade gracefully, smooth not flicker)
and the checklist. MIT licensed, see [LICENSE](LICENSE).
