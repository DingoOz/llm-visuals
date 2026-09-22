# Architecture

llm-visuals is a single Rust binary built on [ratatui](https://ratatui.rs) and
tokio. It has no agents inside the inference server: everything it shows is
polled over HTTP or read from `nvidia-smi`, then derived and smoothed locally.

Every inference server on the machine is watched at once. Each gets a
`ModelSlot` in the frame loop — its own `PerfTracker` and `FadeState` — and
its own HTTP poller. Samples carry the model's PID so the frame loop can route
them; the hardware collectors (`nvidia-smi`, `/proc`) stay shared, because the
GPUs and the disk are shared.

```
 one poller per model, every poll_ms (200 ms)
                ┌──────────────┐                           ┌───────────────┐
 llama-server A ┤ /slots       ├──────────────────────────►│ observe.rs    │
      :8080     │ /metrics     │                           │ LiveStats     │
                │ /experts     │                           │ SpecMetrics   │
                └──────────────┘                           │ ExpertStats   │
                ┌──────────────┐                           │  tagged (pid) │
 llama-server B ┤ /slots …     ├──────────────────────────►│               │
      :8081     └──────────────┘                           └──────┬────────┘
                                                                  │ mpsc channels
                ┌──────────────┐  every 200 ms, shared            │
 nvidia-smi ────┤ --query-gpu  ├──────────────► gpu.rs ───────────┤
                └──────────────┘                GpuStats          │
                ┌──────────────┐  every 400 ms, one pass          │
 /proc ─────────┤ diskstats    ├──────────────► host.rs ──────────┤
 nvidia-smi ────┤ pid/io,stat  │            Vec<(pid,HostSample)> │
                │ dmon -s t    │                                  ▼
                └──────────────┘              ┌─────────────────────────────────┐
                                              │ main.rs event loop  ~30 fps     │
 model_detect.rs ─ /proc + compute apps ─────►│ routes each sample by pid into  │
 gguf.rs ──────── GGUF header ───────────────►│ ModelSlot { perf, fade, live }  │
                                              └───┬──────────┬──────────────────┘
                                                  │          │
                                       perf.rs ◄──┘          └──► fade.rs
                                       rates, TTFT,               attack/release
                                       request log,               smoothing,
                                       MTP acceptance,            expert heat
                                       peak hold          (one of each per model)
                                                  │          │
                                                  ▼          ▼
                                              ┌────────────────────────┐
                                              │ render.rs              │
                                              │ Dashboard {            │
                                              │   models: [ModelView], │
                                              │   focus,               │
                                              │ } → panels             │
                                              └────────────────────────┘
```

## Modules

| Module | Responsibility |
|---|---|
| `main.rs` | CLI parsing, terminal setup, the `ModelSlot` list, spawning one poller per model, routing samples by PID, the frame loop, key handling |
| `config.rs` | clap arguments and `ViewMode` |
| `settings.rs` | per-user config/data paths; loads `settings.json` and places its flags ahead of the command line; the settings screen's fields, keys, validation (re-parsing through clap) and saving |
| `update.rs` | background check of the GitHub release; asks before replacing this binary unless Auto upgrade is on. Source builds are skipped. The download is HTTPS from GitHub only, checked against the published sha256, and swapped in without restarting the session |
| `dblog.rs` | `--log-db` SQLite writer (on by default at the per-user data path) with a size cap |
| `model_detect.rs` | finds inference processes via `nvidia-smi --query-compute-apps` and `/proc`, parses their command lines (model path, port, ctx size, tensor split, spec mode); returns every server found, best first, minus this process and idle daemons |
| `gguf.rs` | reads the GGUF header without loading tensors; maps layers to GPUs from `--tensor-split`; `read_tensor_summary` walks the tensor table and sizes each tensor from the gap to the next offset (no quant type table needed) |
| `observe.rs` | HTTP GET with timeouts; parsers for `/slots`, `/metrics` (Prometheus text) and `/experts` |
| `vllm.rs` | vLLM `/metrics` adapter: reconstructs per-request `LiveStats` from engine-wide Prometheus counters |
| `sglang.rs` | SGLang adapter: `GET /v1/loads?include=all` plus one-shot `/server_info`; optional `sglang:realtime_tokens_total` |
| `gpu.rs` | `nvidia-smi` CSV collector on a `spawn_blocking` thread; smooth random-walk demo GPUs |
| `perf.rs` | turns counter samples into rates with `RateWindow` (sliding window), tracks requests, TTFT, peaks, MTP acceptance, history ring buffers; `Meter` (VU channel with peak hold and auto scale) and `BandwidthStats` for the pipeline view |
| `host.rs` | host counters: `/proc/diskstats`, `/proc/<pid>/io`, `/proc/<pid>/stat` (major faults), `/proc/<pid>/status` (`RssFile`), `/proc/meminfo`, and PCIe rx/tx per GPU from `nvidia-smi dmon -s t -c 1`. The system-wide reads happen once per poll and are shared across every PID, so watching six models costs the same `nvidia-smi` calls as watching one |
| `bandwidth.rs` | `weight_layout` (bytes total / active / CPU-side from the GGUF tensor table and VRAM use) and `assess`, the rule chain that names the bottleneck stage |
| `fade.rs` | time-based smoothing so the UI breathes: fast attack, slow release; per-expert heat that cools exponentially |
| `demo.rs` | synthetic servers (`--demo-models N`, each a prefill → decode → idle loop with its own profile) emitting `LiveStats`, `SpecMetrics` and `ExpertStats` through the same tagged channels; one extra task walks the shared GPUs and host counters from their combined load |
| `colors.rs` | truecolor / 256-colour gating, gradients, palette constants, themes |
| `render.rs` | the `Dashboard` view-model (every `ModelView` plus the focused index) and every panel; the models strip and comparison view walk all of them, the rest use the focused one; gauges, sparklines, big digits, layout rules |
| `pipeline.rs`, `llm/` | attention buffers for the optional Python transformers bridge (`--model <hf id>`) |

## Data contracts

**`/slots`** (always on in llama-server) gives per slot: `n_ctx`, `is_processing`,
`id_task`, `n_prompt_tokens`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache`,
`next_token[0].n_decoded`, `params["speculative.types"]`. Rates are deltas of
these over `RateWindow`s; a request record starts when `id_task` changes or the
slot turns busy, and ends when it turns idle.

**`/metrics`** (needs `--metrics`) gives cumulative
`spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total`,
`spec_decode_num_drafts_total`, `tokens_predicted_total`. Acceptance is the
windowed ratio of the first two.

**`/experts`** (needs the patch in `patches/`) gives per MoE layer the newest 16
routings and a 256-token histogram. See `patches/README.md`.

Each optional endpoint is probed at start and after a rescan; after three
failures that model's poller stops asking, so unpatched or older servers cost
nothing. The judgement is per model: a patched server next to an unpatched one
still gets its expert routing.

**SGLang `/v1/loads?include=all`** is always on. `decode_moments[5]` is
cumulative generated tokens, `decode_moments[0]` is decode/verify steps,
`total_prefill_uncached_tokens` is prefill, `num_used_tokens` is context
occupancy, `memory.weight_gb` / `memory.kv_cache_gb` are the real VRAM split.
There is no completion counter: a request starts on the idle→busy edge or
when `num_used_tokens` drops sharply while still busy. `/server_info` is
fetched once for `context_length` and `speculative_algorithm`. Speculative
acceptance is `generated − steps`, drafted tokens are
`steps × (num_draft_tokens − 1)`. The SGLang poller never runs faster than
400 ms (each GET is a uvicorn access-log line).

**HuggingFace `config.json`** (safetensors dirs, including nested
`text_config`) fills the same architecture fields as a GGUF header, so the
layers and experts panels work for vLLM and SGLang.

A rescan aborts every poller and starts fresh ones for what it finds. Slots
are matched to the new scan by PID, so a model that is still running keeps its
counters, history and request log, and one that has gone away stops being
polled immediately.

**Host counters** are cumulative, so `BandwidthStats::observe_host` takes
deltas over the poll interval. `nvidia-smi dmon` is already a rate (MB/s
over its own 1 s window) but occasionally prints a nonsense sample; anything
above the PCIe link cap (`pcie_link_mb_s(gen, width)`) is dropped.

**Weight streams** are estimates, labelled as such on screen:
`bytes_per_step = total − embedding − experts × (1 − used/total)`;
`cpu_bytes = file − Σ min(file × split share, VRAM used)`; RAM GB/s =
`cpu_bytes × active fraction × steps/s`, VRAM GB/s = the rest. Steps/s is
`spec.steps_per_sec` when `/metrics` is served, else decode tok/s; during
prefill it is `prefill tok/s ÷ ubatch`.

## Timing model

- Pollers run on their own tasks and push into bounded channels.
- The frame loop drains every channel, feeds `PerfTracker` and `FadeState`,
  builds a `Dashboard` of borrowed references, and renders. It sleeps ~33 ms.
- `FadeState::tick` uses wall-clock `dt`, so smoothing looks the same at any
  frame rate. Attack ≈ 180 ms, release ≈ 2.4 s for layer tiles; expert heat
  cools with τ = 0.6 s.
- `RateWindow` stores `(interval start, interval end, tokens)` so a rate is
  tokens over real elapsed time, including the first interval, and drops to
  zero within one window of the counter stopping.

## Rendering rules

- Colour comes from `colors::rgb`, which quantises to the xterm-256 cube when
  the terminal is not truecolor (auto-detected from `COLORTERM`, `TERM`,
  `TERM_PROGRAM`, VTE/Konsole markers; override with `--color`).
- Panels use rounded borders with an accent-coloured left title. A right-aligned
  border title is attached only if it cannot collide with the left one
  (`with_right`); otherwise the information moves into the body or is shed.
- Gauges have half-cell precision and an optional white peak-hold notch.
  The pipeline view's vertical meters (`vmeter`) use eighth-block precision,
  green at the foot to red at the top, with a `▔` peak line that holds 1.2 s
  and then falls at 40 % of scale per second.
- Sparklines are right-aligned in time (newest at the right edge) and coloured
  by level with a gradient.
- Layout sheds the models strip first (it needs 14 rows of body beneath it),
  then requests below 22 body rows and context/MTP below 16; the GPU line
  sheds PCIe, fan, clock, temperature before shrinking its gauge, and the key
  row shortens its own labels before dropping keys from the end.
- The models strip gives each row a fixed budget — index, name, phase and rate
  always survive — and spends what is left on history, then context, then
  VRAM, so every row's columns line up at any width.

## Testing

`cargo test` runs unit tests for every parser (real `nvidia-smi`, `/slots`,
`/metrics` and `/experts` samples), the rate window, request lifecycle, peak
hold, expert heat, colour quantisation, sparkline shapes and formatting. UI
layout is checked by running the binary in tmux and capturing panes; the
`docs/*.svg` screenshots are produced that way.
