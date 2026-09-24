# Error Log

## Summary

22 entries (as of 2026-09-24). Recurring themes:

- **Optional or missing telemetry treated as a real value** (Logic, most common): a missing counter read as 0, record close gated on optional TTFT, unknown ctx rendered as full, model ownership derived from an unknown weight estimate. Rule of thumb: keep "unknown" distinct from zero and never gate state or ownership on an optional measurement.
- **Under-discriminating matches when resolving processes/devices**: docker-proxy matched by IP only, comm-name gating, xe fans keyed by a constant path component, env GPU masks merged with host indices. Match on every discriminating field and prefer authoritative (driver/host) sources over inferred ones.
- **NVML/driver API semantics differing from the CLI** (API Misuse): power limit, reserved VRAM, NOT_SUPPORTED handling, session lifetime.
- **Hot-path cost**: per-device queries for filtered GPUs, blocking sleeps inside the 200 ms GPU poll.

### vLLM positional arg parser consumed argv[0] — 2026-09-16

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** A positional-argument catch-all in a cmdline parser that accepts "any bare token" also matches argv[0] (always a full binary path in /proc cmdline) and bare values of unhandled flags.
- **Root cause:** The vLLM catch-all only checked "doesn't start with `-` and path not yet set", which argv[0] satisfies before any subcommand token is seen.
- **Fix applied:** Positional capture now requires a preceding `serve`/`api-server` token and a path-shaped value (contains `/` or a weights extension).
- **Prevention rule:** When parsing /proc cmdlines, never let a positional catch-all run before the subcommand token; add a test with the full binary path as argv[0].

### vLLM phantom filter keyed on comm == "vllm" — 2026-09-16

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** Classifying a process by requiring an exact comm name, when the same server can appear as `python3`, `pt_main_thread`, or a full executable path depending on launch form and data source.
- **Root cause:** The filter treated "comm != 'vllm' and no --port" as a phantom, dropping real servers launched via the python module form or without an explicit port.
- **Fix applied:** Classify by argv[0] basename (`vllm`) or `vllm.entrypoints` in the cmdline; name prefixes only for known helper/client patterns.
- **Prevention rule:** Never gate process classification on an exact comm string; test with the module-invocation and default-port launch forms.

### Windowed rate cleared on the only poll that carried the delta — 2026-09-16

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/perf.rs`
- **Pattern:** For counters that only move on a terminal event (vLLM tokens move at completion), clearing/resetting the measurement window on that same event reads the value as permanently zero.
- **Root cause:** `decode_win.clear()` on `!processing` ran before the rate was read, and only prefill received the histogram-based clamp.
- **Fix applied:** Symmetric decode clamp from the server ITL sum (`(decoded − 1) / itl_sum`) on the completion poll; test asserts `peak_decode_tps`.
- **Prevention rule:** When adding a rate fallback for completion-only counters, apply it to every rate derived from those counters and assert each in the test.

### Record close gated on an optional measurement — 2026-09-16

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/perf.rs`
- **Pattern:** Setting a record's terminal state (`ended`) only inside a branch that also requires an optional measurement (TTFT > 0), leaving the record open forever when the measurement is absent.
- **Root cause:** The backdate logic and the `ended` stamp were fused in one conditional.
- **Fix applied:** `ended` is stamped unconditionally; only the backdate stays gated on a usable TTFT.
- **Prevention rule:** State transitions must never depend on optional telemetry; gate only the enrichment, not the transition.

### Poll loop with no failure path — 2026-09-16

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/main.rs`
- **Pattern:** A polling loop whose failure branch is empty freezes downstream consumers on the last accepted sample and keeps polling a deterministically-failing target at full rate.
- **Root cause:** The vLLM loop had `if let Some(..)` with no else; the llama.cpp path's miss-counter policy was not mirrored.
- **Fix applied:** Miss counter: after 3 consecutive failures send an idle sample (UI decays) and back the poll interval off to ≥2 s.
- **Prevention rule:** Every poll loop needs an explicit failure branch: decay the consumer's state and back off; mirror the existing loop's policy when adding a sibling.

### docker-proxy match by container IP only — 2026-09-16

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** Resolving a container port mapping by matching only the container IP returns an arbitrary mapping when the container publishes several ports (one docker-proxy per mapping, same IP).
- **Root cause:** `-container-port` was not captured or compared against the server's own port.
- **Fix applied:** The match now also requires `-container-port == cmdline_port.unwrap_or(8000)`.
- **Prevention rule:** When resolving via /proc process scans, match on every discriminating field available, not just the first sufficient-looking one.

### Unknown context length rendered as 100% full — 2026-09-16

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/main.rs`, `src/model_detect.rs`
- **Pattern:** Substituting `.max(1)` for an unknown denominator turns "unknown" into "completely full" in every ratio-based display.
- **Root cause:** vLLM safetensors servers without `--max-model-len` had no ctx source (the GGUF fallback needs a .gguf file), leaving ctx_max = 0.
- **Fix applied:** Read `max_position_embeddings` from the weights dir's config.json — exactly vLLM's own default for `max-model-len`.
- **Prevention rule:** Every engine-specific detection path needs its own source for each field the renderer divides by; grep for `.max(1)` on the field when adding a new engine.

### Test asserting a duplicated copy of production logic — 2026-09-16

- **Severity:** Low
- **Category:** Convention
- **File(s):** `src/vllm.rs`
- **Pattern:** A unit test exercising a verbatim in-test copy of a production predicate, which stays green when the shipped code drifts.
- **Root cause:** The cross-wiring guard was inlined in an async fn and copy-pasted into the test module for testability.
- **Fix applied:** Extracted `fn cross_wired(..)`, called from `poll_vllm` and tested directly; deleted the duplicate.
- **Prevention rule:** If logic must be copied to be testable, extract it instead — a test may only assert code the binary actually runs.

### nvidia-smi compute-app rows dropped on "[N/A]" memory — 2026-09-17

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** Discarding a whole CSV record from an external tool because one informational column fails to parse, when tools print placeholders like `[N/A]` / `[Not Supported]` on some drivers or platforms.
- **Root cause:** `used_gpu_memory` was parsed with `continue` on error; Windows (WDDM) drivers always report `[N/A]`, so every GPU process vanished from detection.
- **Fix applied:** Parse the column with `unwrap_or(0)` so the process is kept with unknown memory.
- **Prevention rule:** When parsing `nvidia-smi` (or similar) CSV, only skip a row when its identity columns (pid, uuid, index) fail to parse; default non-key numeric columns.

### Discover tests assumed an empty host process table — 2026-09-20

- **Severity:** Low
- **Category:** Logic
- **File(s):** `src/main.rs`
- **Pattern:** An integration test of `discover()` asserting `models.is_empty()` or `models.len() == 1` for an explicit `--endpoint`, while production still scans the host process table and can attach extra real servers.
- **Root cause:** Mock-server tests were written on a machine with no llama-server running, so process detection returned nothing and the assertions only covered the mock.
- **Fix applied:** Assert the explicit endpoint is present (and error text for failures) without requiring it to be the only detected model.
- **Prevention rule:** Tests that call `discover()` must select the model under test by port/name; never assert the whole result set is empty or length 1 unless process scanning is stubbed.

### Missing counter treated as a real zero — 2026-09-22

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/observe.rs`, `src/main.rs`
- **Pattern:** `unwrap_or(0)` on a JSON field that an upstream server may omit, so "not reported" and "reported as zero" become the same value and no fallback can tell them apart.
- **Root cause:** Recent llama.cpp dev builds leave `n_decoded` out of `/slots` during generation. The parser stored 0, the rate window saw no delta, and finished requests never accumulated decode tokens.
- **Fix applied:** `decoded_present` records whether the field was a number. When it was not, the poller fills `decoded` from a per-request anchor on `tokens_predicted_total`.
- **Prevention rule:** When a sampled counter is optional, store presence separately from the value. Do not use `unwrap_or(0)` on it if zero is also a meaningful sample.

### Poll target hardcoded to loopback — 2026-09-22

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/main.rs`, `src/model_detect.rs`, `src/observe.rs`, `src/sglang.rs`, `src/vllm.rs`
- **Pattern:** A detector records a server's listen address and every poller then dials a hardcoded `127.0.0.1` instead.
- **Root cause:** `--host <lan-ip>` was parsed nowhere, so a server that does not bind loopback was listed from the process table and then polled at an address that refuses the connection.
- **Fix applied:** `DetectedModel.host` comes from `--host`. Wildcard binds (`0.0.0.0`, `::`, empty) still dial `127.0.0.1`. The local port probe also tries a few non-loopback addresses from the machine's own fib_trie.
- **Prevention rule:** Any new poll or probe takes the detected host. A wildcard listen address must be dialed as loopback; a concrete address must be dialed as itself. Add a cmdline test for both.

### NVML device dropped when one optional query returns NOT_SUPPORTED — 2026-09-23

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/nvml.rs`
- **Pattern:** `continue`-ing past a whole device/record when one *informational* field of a driver or CLI query fails, so the entity silently disappears from an aggregate. Recurrence of the 2026-09-17 `[N/A]` entry in a different API (FFI instead of CSV).
- **Root cause:** `collect_stats` skipped the device when `nvmlDeviceGetMemoryInfo` or `nvmlDeviceGetUtilizationRates` was not `NVML_SUCCESS`; both return `NVML_ERROR_NOT_SUPPORTED` on MIG-enabled devices. The panel showed one fewer GPU and `total_power_w` (and the `tok/J` derived from it) were understated with no error.
- **Fix applied:** Both fields default to zero like every other optional field; the fallback signal moved to "no device reported non-zero VRAM", matching `collect_amd`.
- **Prevention rule:** Only drop a device/record when its *identity* lookup fails (handle, index, pid). Default every telemetry field; signal an unusable backend from an all-devices-unreadable test, never by shrinking the list.

### Stateful NVML session cached for process lifetime could not self-heal — 2026-09-23

- **Severity:** High
- **Category:** API Misuse
- **File(s):** `src/gpu.rs`, `src/nvml.rs`
- **Pattern:** Replacing a stateless per-poll collector (subprocess, HTTP request) with a long-lived stateful session probed once at startup, without a recovery path. The old design recovered from external restarts for free; the new one wedges permanently.
- **Root cause:** `GpuBackend::detect` ran once and the `Arc<NvmlSession>` was reused forever. After a driver reload, package upgrade, GPU reset or Xid, NVML returns `UNINITIALIZED`/`GPU_IS_LOST` on every subsequent call, where the `nvidia-smi` path it replaced simply worked again on the next poll.
- **Fix applied:** `NvmlSession::reinit` (shutdown + `nvmlInit_v2`), retried once from `device_count()`. Handles are already re-fetched by index each poll, and both `gpu.rs` and `host.rs` share the `Arc`, so both consumers recover.
- **Prevention rule:** When swapping a per-call collector for a persistent session, ask what happens when the far side restarts — and re-establish the session on the first error rather than caching failure.

### Hardware test asserted on conditions production treats as fallback — 2026-09-23

- **Severity:** Medium
- **Category:** Convention
- **File(s):** `src/nvml.rs`
- **Pattern:** A hardware-gated test whose skip guard is narrower than production's fallback condition, so it fails on machines the product handles correctly.
- **Root cause:** `test_nvml_lifecycle_and_stats` skipped only when `NvmlSession::new()` returned `None`, then `.expect()`ed stats and asserted non-empty. A container with `libnvidia-ml.so.1` but no `/dev/nvidia*` (count 0) or an all-MIG host (`Err`) both panic, while `GpuBackend::detect` correctly falls through to `nvidia-smi`.
- **Fix applied:** Every condition that makes production fall back — `None`, `Err`, zero devices — is now a `println!` + `return`.
- **Prevention rule:** A hardware test's skip guard must cover exactly the set of states the production fallback tolerates. Enumerate them from the production branch, not from the dev machine.

### NVML power limit not the same quantity as nvidia-smi power.limit — 2026-09-23

- **Severity:** Low
- **Category:** API Misuse
- **File(s):** `src/nvml.rs`
- **Pattern:** Two backends for the same displayed metric reading different underlying quantities, so a gauge's meaning changes with a `--flag` that is meant to be transparent.
- **Root cause:** The NVML path preferred `nvmlDeviceGetEnforcedPowerLimit` while the CSV path parses `power.limit`, which is the *management* limit (`nvidia-smi` exposes the enforced one separately as `enforced.power.limit`). On a thermally capped card the two differ, so `power_frac()`'s full scale differed with and without `--no-nvml`.
- **Fix applied:** Prefer `nvmlDeviceGetPowerManagementLimit`, falling back to the enforced limit only if the symbol is missing.
- **Prevention rule:** When adding a second backend for an existing metric, map each field to the *exact* counter the original used — matching names are not matching semantics.

### Sampled per-device driver calls issued for GPUs the panel filters out — 2026-09-23

- **Severity:** Low
- **Category:** Other
- **File(s):** `src/nvml.rs`, `src/host.rs`, `src/main.rs`
- **Pattern:** A collector iterating every device when a display filter (`--gpu`) already narrows what is shown, where each call has a fixed driver-side sampling cost.
- **Root cause:** `collect_pcie_throughput` looped `0..count` with two `nvmlDeviceGetPcieThroughput` calls per device; the driver samples that counter over ~20ms per call, so an 8-GPU host spent ~320ms inside a 400ms host poll and jittered the `dt` that `perf::observe_host` divides rates by.
- **Fix applied:** `collect_pcie_throughput(&self, filter: &[usize])`, threaded from `args.gpu_indices()` through `HostMonitor::new`.
- **Prevention rule:** Push the display filter down to the collector whenever a per-item query blocks; pass it as a parameter rather than filtering the result.

### NVML VRAM used included the driver's reserved carve-out — 2026-09-23

- **Severity:** Medium
- **Category:** API Misuse
- **File(s):** `src/nvml.rs`
- **Pattern:** Same as the power-limit entry: a second backend reading a counter with the same name but different semantics.
- **Root cause:** `nvmlDeviceGetMemoryInfo` (v1) folds the driver's reserved memory into `used`; `nvidia-smi`'s `memory.used` comes from the v2 struct, which reports `reserved` separately. On an idle RTX 5060 Ti NVML said 494 MB used against `nvidia-smi`'s 33 MB (RTX 3070: 364 vs 15), which leaks into the VRAM gauge and the weights/KV split. `bytes_to_mb` also truncated where `nvidia-smi` rounds, reading 1 MB low.
- **Fix applied:** Prefer `nvmlDeviceGetMemoryInfo_v2` (R510+) with `version = NVML_STRUCT_VERSION(Memory, 2)`, falling back to v1. Round to the nearest MiB. Verified on hardware: both backends now agree to within 1 MB on every field.
- **Prevention rule:** Verify a new backend against the old one on real hardware, field by field, before relying on "matching" names.

### xe hwmon BDF read from a constant path component — 2026-09-24

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/gpu.rs`
- **Pattern:** Taking `file_name()` of a sysfs symlink path (`hwmonX/device`) expecting the link target's name; it always returns the link's own name, so every entry shares one sort key and the order silently falls back to the next tuple field.
- **Root cause:** `path.join("device").file_name()` never follows the symlink, so all cards keyed as `"device"` and fans were mapped to GPUs in RPM order.
- **Fix applied:** `fs::canonicalize(path.join("device"))` before taking the BDF; sysfs read split into `xe_fan_rpm_raw(root)` with a symlink-based test where hwmon, BDF and RPM orders all disagree.
- **Prevention rule:** For sysfs device identity, canonicalize the `device` link; test ordering with fixtures whose alternative sort keys disagree.

### Blocking retry sleeps inside the GPU poll — 2026-09-24

- **Severity:** Medium
- **Category:** Other
- **File(s):** `src/gpu.rs`
- **Pattern:** A read-retry loop that sleeps whenever a sensor reads 0, where 0 is also a legitimate steady state, so the steady state pays the maximum delay on every poll.
- **Root cause:** The xe tachometer burst-read slept 3×120 ms per card when fans were stopped (normal at idle), stretching the 200 ms poll to ~1.4 s on 4 cards.
- **Fix applied:** Single read per poll; the existing 20 s hold bridges the sensor's update cadence.
- **Prevention rule:** No sleeps in `GpuBackend::collect` paths; smooth intermittent sensors across polls with state, not within one poll.

### Model ownership derived from the weight-size estimate — 2026-09-24

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/main.rs`, `src/render.rs`
- **Pattern:** Deriving a placement fact (which GPUs a model occupies) from an optional size estimate, so an unknown size reads as "not here".
- **Root cause:** `model_owned` was `weight_frac > 0`; with no weight size (common on Intel: no per-process memory) the focused model's own cards were labelled "other server".
- **Fix applied:** `model_owned` is set from the placement `share > 0` in `fade_sample_from_live`.
- **Prevention rule:** Placement comes from `gpu_indices`/tensor split only; never infer it from byte estimates.

### Environment GPU mask merged with driver-reported host indices — 2026-09-24

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** Merging GPU indices from a process's env (`CUDA_VISIBLE_DEVICES`, `ZE_AFFINITY_MASK`), which may be relative to a container's device set, with driver-reported host indices.
- **Root cause:** Env affinity seeded `gpu_indices` before the compute-app worker merge, so a container mask `0` plus host GPU `2` produced `[0, 2]`. Mask forms `0,1` and tile syntax `2.0` also parsed to nothing.
- **Fix applied:** Env affinity applies only when no driver placement exists; both variables parse as comma lists with the tile suffix stripped (`parse_gpu_affinity`, tested).
- **Prevention rule:** Driver-reported indices are authoritative; inferred placement is a fallback only, never merged with them.

