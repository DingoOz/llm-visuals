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
  NVML / SMI ───┤ telemetry    ├──────────────► gpu.rs ───────────┤
  or amdgpu     └──────────────┘                GpuStats          │
                ┌──────────────┐  every 400 ms, one pass          │
  /proc ────────┤ diskstats    ├──────────────► host.rs ──────────┤
  NVML / dmon ──┤ pid/io,stat  │            Vec<(pid,HostSample)> │
                │ pcie rx/tx   │                                  ▼
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
| `dblog.rs` | `--log-db` SQLite writer (on by default at the per-user data path) with a size cap; the `bench_runs` / `bench_results` tables and their read-back for `--bench-report` |
| `bench.rs` | `--bench`: the plan (lengths that fit the server's context), deterministic prompts with a per-request unique first word, the streamed `/v1/completions` client timing (TTFT, decode), tokens-per-word calibration from the server's `usage` / `timings` / `/tokenize`, per-length summaries, and the text tables printed on exit and by `--bench-report` |
| `regress.rs` | the regression flag: reads the last hour and the baseline window of decode samples from the log on a blocking thread (rowid binary search as the time index, baseline cached for an hour), keeps spans logged with the model's current GPU set, and compares the two per context range |
| `model_detect.rs` | finds inference processes via `nvidia-smi --query-compute-apps` and `/proc`, parses their command lines (model path, port, ctx size, tensor split, spec mode); returns every server found, best first, minus this process and idle daemons |
| `vision.rs` | vision encoder detection and placement: llama.cpp projector flags and environment, the GPU runtime from `/proc/<pid>/maps`, and CUDA ordinal → host card via a `--cuda-bus-ids` child process that queries `libcuda` under the server's `CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER` |
| `gguf.rs` | reads the GGUF header without loading tensors; maps layers to GPUs from `--tensor-split`; `read_tensor_table` walks the tensor table and sizes each tensor from the gap to the next offset (no quant type table needed), `read_tensor_summary` totals it |
| `offload.rs` | llama.cpp's buffer-type overrides (`-ot`, `--cpu-moe`, `--n-cpu-moe`, `--n-cpu-ffn`, their `LLAMA_ARG_*` variables) parsed into regex rules and applied to the tensor table at detection: bytes in RAM (and read per step), bytes pinned to each card, and per layer the share in RAM and the card it is pinned to |
| `observe.rs` | HTTP GET with timeouts; `HttpStream`, a POST whose body is read as it arrives (chunked transfer decoding included), for the benchmark; parsers for `/slots`, `/metrics` (Prometheus text) and `/experts` |
| `vllm.rs` | vLLM `/metrics` adapter: reconstructs per-request `LiveStats` from engine-wide Prometheus counters |
| `ollama.rs` | Ollama: `GET /api/ps` (tag, size / size_vram, context, expiry), the manifest walk that names a runner's blob, `OLLAMA_HOST` parsing, the store lookup that finds a daemon slot's GGUF |
| `sglang.rs` | SGLang adapter: `GET /v1/loads?include=all` plus one-shot `/server_info`; optional `sglang:realtime_tokens_total` |
| `strata.rs` | Strata and Project Maya adapter: JSON `GET /metrics` (live request, finished-request totals, MTP drafts; Maya's `tiers` expert-tier counters and `gpu_map` VRAM split), `/health`, the server's `--config` JSON (Maya's pack and cards), and which processes are the server, its native engine and Maya's vision helper |
| `gpu.rs` | in-process NVML telemetry collector (falling back to `nvidia-smi` CSV) or Linux amdgpu sysfs on a `spawn_blocking` thread; smooth random-walk demo GPUs |
| `nvml.rs` | dynamically loads `nvml.dll` or `libnvidia-ml.so`; maintains a persistent in-process session for GPU metrics and PCIe throughput without subprocess overhead |
| `perf.rs` | turns counter samples into rates with `RateWindow` (sliding window), tracks requests, TTFT, peaks, MTP acceptance, history ring buffers, the clocks-event reason hold; `Meter` (VU channel with peak hold and auto scale) and `BandwidthStats` for the pipeline view |
| `energy.rs` | GPU energy: summed board power integrated over time (trapezoid, gaps over 5 s skipped) for the box, plus each model's share — energy and tokens over the intervals it was the only model working — and the `--energy-price` tariff |
| `host.rs` | host counters: `/proc/diskstats`, `/proc/<pid>/io`, `/proc/<pid>/stat` (major faults), `/proc/<pid>/status` (`RssFile`), `/proc/meminfo`, and PCIe rx/tx per GPU via NVML `nvmlDeviceGetPcieThroughput` (with `nvidia-smi dmon` fallback). In-process NVML eliminates the repeated CLI subprocess spawns on the NVML path; the `dmon` fallback still spawns one per poll |
| `bandwidth.rs` | `weight_layout` (bytes total / active / CPU-side from the GGUF tensor table and VRAM use) and `assess`, the rule chain that names the bottleneck stage |
| `fade.rs` | time-based smoothing so the UI breathes: fast attack, slow release; per-expert heat that cools exponentially; per-expert usage (the server's hit counts as a share of the busiest expert's) eased for the landscape |
| `landscape.rs` | the expert landscape (`e`), free of terminal types: a `Camera` that rests at the four isometric corners, the projection, `Grid` (experts and layers grouped to what the canvas resolves) and a small rasteriser — boxes in painter's order, supersampled, with a blurred glow — that returns an RGB image `render.rs` paints with half blocks |
| `balance.rs` | router load balance for the balance view (`u`), free of terminal types: per layer, from the `/experts` hit counts, effective experts (exp entropy), top-k share, a ranked-hits curve, and which layers stand out against the median layer |
| `replay.rs` | `--replay`: reads a `--log-db` file read-only, splits it into sessions on five-minute holes, and turns its rows back into the `LiveStats` and GPU frames a live poller would have produced, stamped on a clock that runs in logged seconds; pause, seek (with a minute of warm-up) and speed |
| `demo.rs` | synthetic servers (`--demo-models N`, each a prefill → decode → idle loop with its own profile) emitting `LiveStats`, `SpecMetrics` and `ExpertStats` through the same tagged channels; one extra task walks the shared GPUs and host counters from their combined load |
| `alerts.rs` | the alert conditions (VRAM growth past the level each card settled at, GPU / memory hot, slow decode against the model's own context-matched baseline, server gone), each a hold → raise → hysteresis → clear episode with a per-subject announcement cooldown; free of terminal types, fed by the frame loop four times a second; plus the desktop notifier (`notify-send` / `osascript`) |
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

**Ollama `/api/ps`** lists the loaded models with `size` (bytes resident),
`size_vram`, `context_length`, `expires_at` and `details`. Current Ollama
serves each model from a bundled llama-server, which is polled through
`/slots` like any other; its poller also asks the daemon every 5 s and puts
the matching entry on every `LiveStats` (`ollama`). A runner's entry is the
tag whose manifest names its blob, the slot is renamed to the tag the daemon
reports, and `size − size_vram` replaces the tensor-split estimate of
CPU-side bytes. While `/slots` is silent the daemon is asked every 2 s instead:
the runner exits with its model, and a sample saying so marks the slot
evicted. A daemon slot or a pre-llama-server `ollama runner` has no counters,
so its poller sends `/api/ps` alone every 2 s.

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
`cpu_bytes × active fraction × steps/s`, VRAM GB/s = the rest. When
llama.cpp's override rules place tensors (`offload.rs`), the split share
covers only the tensors no rule placed: `cpu_bytes = file − Σ min(unplaced
× split share + pinned to that card, VRAM used)`, so the rules' RAM tensors
are always CPU-side. Per step the RAM reads those tensors at their own
active bytes (routed experts × used/total) plus any spill at the rest of the
model's active fraction. Steps/s is
`spec.steps_per_sec` when `/metrics` is served, else decode tok/s; during
prefill it is `prefill tok/s ÷ ubatch`.

**Peak memory bandwidth** is driver arithmetic, not an estimate:
`bus width / 8 × max memory clock × 2` from NVML
(`nvmlDeviceGetMemoryBusWidth`, `nvmlDeviceGetMaxClockInfo(MEM)`), read once
per card. The VRAM stream's ceiling combines the cards that hold weights:
`Σ bytes_i / Σ (bytes_i / peak_i)` under llama.cpp's layer split, where the
cards read their shares one after another, and `Σ bytes_i / max (bytes_i /
peak_i)` under a parallel one (tensor-parallel safetensors servers,
`--split-mode row`). It is unknown unless every such card has a peak.

**Replay** (`--replay`) replaces every poller and hardware collector with
one source driven from the frame loop rather than its own task, because
seeking has to restart the trackers in step with the samples:

```
 llm.db (read-only) ─► replay.rs ─ step(wall dt × speed) ─► Event::Reset
   model_samples       one LiveStats per model per 0.2 s     Event::Live{ts,pid}
   requests            of logged time; GPU rows at their ts  Event::Gpu{ts}
   gpu_samples                         │
                                       ▼
                     main.rs: PerfTracker::observe(_, replay.instant(ts))
```

Sessions are found without a scan of the log: rows are appended in time
order and trimmed oldest first, so rowid order is time order and a binary
search over rowids stands in for the index `ts` does not have. The newest
session is read by looking back from the end through a window that grows
until the session's start is inside it.

`replay.instant(ts)` maps logged time onto an `Instant` one logged second per
second at any speed, so rate windows, peak holds and request durations are
computed as they were live. Within a request, tokens follow the interpolated
session counters, scaled to the request row's totals; its start, first token
and end come from that row. `fade.rs` keeps wall-clock time: smoothing is a
display effect, not part of the record.

**Benchmark (`--bench`)** is the one place the dashboard sends requests
rather than only reading. `bench::run` is a tokio task that posts streamed
completions to the focused server and reports `BenchEvent`s (plan, progress,
one result per request) on a channel the frame loop drains. The frame loop
owns the `BenchSession` the `B` table draws, writes the run and its results
to the log, and gives each result the dashboard's own row for the request:
it notes the slot's `PerfTracker::finished` when a request is sent, and the
first request row closed after that is the one (waited for up to 3 s, else
left empty). The task paces requests one second apart for that reason.

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
  sheds PCIe, fan, clock, temperature before shrinking its gauge (whole
  items, never cut at the panel edge), keeps a clocks-event badge longest
  and lets it take cells from the gauge, and the key row shortens its own
  labels before dropping keys from the end.
- Clocks-event reasons are held per bit for 1.5 s after the driver last
  reported them (`perf::ReasonHold`), so a one-poll power cap is readable;
  the hold stretches a reported reason and never adds one.
- The models strip gives each row a fixed budget — index, name, phase and rate
  always survive — and spends what is left on history, then context, then
  VRAM, so every row's columns line up at any width.

## Testing

`cargo test` runs unit tests for every parser (real `nvidia-smi`, `/slots`,
`/metrics` and `/experts` samples), the rate window, request lifecycle, peak
hold, expert heat, colour quantisation, sparkline shapes and formatting. UI
layout is checked by running the binary in tmux and capturing panes; the
`docs/*.svg` screenshots are produced that way.
