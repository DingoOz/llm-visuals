# Stats export: `llm-visuals.stats/1`

llm-visuals can publish its live numbers as JSON for other programs: status
bars, desk gadgets, scripts. There are two outputs, and both can run at once.
Both work with `--demo` and `--replay` (the `source` field says which).

| Flag | Output |
|------|--------|
| `--export-json PATH` | One JSON object in a file, replaced on every export. |
| `--export-sock PATH` | A Unix socket (Linux and macOS). One JSON object per line (NDJSON) on every export, to every connected client. |

Neither is on by default, and either one makes llm-visuals headless (see
below). On other platforms `--export-sock` stops the
dashboard at startup with an error.

## Headless by default

With `--export-json` or `--export-sock`, llm-visuals runs as a daemon: it
does not start the terminal interface, reads no keys, and needs no tty (it
works with `stdin` from `/dev/null`, under `nohup` or under systemd). Add
`--tui` to keep the dashboard on screen while it exports.

In headless mode:

- Nothing is written to stdout.
- stderr gets one line per event, in the form `llm-visuals: <message>`,
  without colour codes. The events are: start (version, source, outputs,
  poll interval); a model attached or detached (name, engine, pid,
  endpoint); `no inference server found, waiting` (once); a socket client
  connected or disconnected (at most one line per second); the first error
  of each output, and its recovery; each change of the message the
  dashboard would show in its status line (identical messages are not
  repeated); the end. Nothing is logged per export.
- Servers are looked for again every 5 seconds while none is attached, and
  every 30 seconds after that. This replaces the `r` key. A server that
  starts after llm-visuals is attached, and one that stops is detached.
- `--log-db` keeps logging. Alerts are off.
- SIGINT and SIGTERM end the program normally: the socket file is removed
  and the exit code is 0.
- With `--replay`, the log is played into the export at its own speed. At
  the end of the log the program says so and exits with code 0.
- `--bench` and `--model` (the Python bridge) need the dashboard. With an
  export flag they stop at startup with an error unless `--tui` is given.
- A startup error (for example a socket path that cannot be bound) is one
  `llm-visuals: ...` line and exit code 1.

### Example: systemd user service

Save as `~/.config/systemd/user/llm-visuals-export@.service`. The instance
name (`%i`) names the output files.

```ini
[Unit]
Description=llm-visuals stats export (%i)

[Service]
ExecStart=%h/.cargo/bin/llm-visuals --export-json /run/user/%U/llmv-%i.json --export-sock /run/user/%U/llmv-%i.sock
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

To also serve the network, add `--export-listen 0.0.0.0:9464
--export-token-file %h/.config/llm-visuals/token` to `ExecStart` (see
Network access).

Then run `systemctl --user enable --now llm-visuals-export@box.service`.
The file is `/run/user/<uid>/llmv-box.json`. Errors and events are in
`journalctl --user -u llm-visuals-export@box`. Change `ExecStart` to the
place where the binary is installed.

## Network access

`--export-listen HOST:PORT` serves the export over HTTP, so a client on
another machine can follow a headless daemon. Examples: `127.0.0.1:9464`,
`0.0.0.0:9464`, `[::]:9464`. It is off by default, and it makes llm-visuals
headless like the other export flags.

| Route | Reply |
|-------|-------|
| `GET /stats` | The latest export as one JSON object (`application/json`, `Cache-Control: no-store`). `503` until the first export. |
| `GET /stream` | NDJSON (`application/x-ndjson`, chunked): the latest export at once, then one line per export, the same lines as `--export-sock`, heartbeat included. It ends with the closing chunk when the daemon stops. |
| `GET /health` | `200`, `ok`. Needs no token. |

Query strings are ignored. Other paths give `404`, other methods `405`.
HTTP/1.0 and HTTP/1.1 are accepted. Every reply closes the connection.
A request head over 8 KiB gets `431`; a head that takes over 5 seconds gets
`408`. At most 16 connections are served at once; the next gets `503`.
A `/stream` client that falls 32 exports behind, or does not take data for
5 seconds, is dropped. The exporter never waits for a client.

**Token.** `--export-token-file PATH` names a file with a bearer token (white
space around it is removed; an empty file is a startup error). With it,
`/stats` and `/stream` need `Authorization: Bearer <token>`. Without it,
they get `401` with `WWW-Authenticate: Bearer`. Without a token file, anyone
who can connect can read.

**No CORS, and a Host check.** Replies never carry `Access-Control-Allow-*`
headers: a web page cannot read the export, the client is meant to be a
native program. Without a token, the `Host` header is also checked, against
DNS rebinding (a web page reaching this address under another name). It must
be, with or without a port:

- an IP address, or `localhost`;
- the host part of `--export-listen`;
- this machine's hostname exactly as the system gives it (`/etc/hostname`,
  else `$HOSTNAME`), and the first label of it: on a host named
  `rack7-gpubox2.example.lan`, `rack7-gpubox2.example.lan` and `rack7-gpubox2` pass (any case);
- a name given with `--export-allow-host NAME` (repeatable; exact match, any
  case): the DNS names clients really use, aliases, a reverse proxy.

All matches are exact: there is no suffix or wildcard, so `rack7-gpubox2.evil.com`
and `rack7-gpubox2.example.org` get `403`. If clients reach the daemon by a
name that is not the hostname (a name from another domain, a CNAME), add it
with `--export-allow-host`. For the systemd unit in `contrib/systemd`, put it
in `~/.config/llm-visuals/export.env`:

```text
LLMV_EXPORT_ARGS=--export-allow-host gpubox2.example.org --export-allow-host gpubox2.lan
```

Any other name gets `403`, and a missing `Host` on HTTP/1.1 gets `400`. The
start line lists the accepted names when there is no token. With a token, the
`Host` header is not checked.

**No TLS.** The token and the data cross the network in clear text. Use a
LAN you trust or a VPN, or keep the listener on loopback and use an SSH
tunnel: `ssh -L 9464:127.0.0.1:9464 host`. When the address is not loopback
and there is no token, the start line is followed by a warning.

In the headless log, each `/stream` client gives a `connected` and a
`disconnected` line (with the number of clients), and a `401` gives one line
per client address per minute.

```sh
curl http://host:9464/health
curl -H "Authorization: Bearer $(cat token)" http://host:9464/stats
curl -N -H "Authorization: Bearer $(cat token)" http://host:9464/stream
```

## Cadence

- One export per poll, at most once per `--poll-ms` (default 200 ms, so about 5 per second).
- A heartbeat: an export is sent at least once per second, even when no
  value changes, and even when the screen is idle and nothing is drawn.
  If `--poll-ms` is longer than one second, the heartbeat sets the pace.
- `seq` goes up by one on every export. A consumer should treat a file whose
  `ts` (or modification time) is older than 2 seconds as "no signal".

## Atomicity and failures

- The file is written to a new temporary file in the same directory
  (`.NAME.PID.N.tmp`, created so that it never follows a link or reuses an
  existing file), then renamed to `PATH`. A reader sees the old file or the new file, never a part of one.
- The socket never blocks the dashboard. Clients get non-blocking writes and
  a pending buffer of 256 KiB each. A client that does not read, or that
  fails, is dropped; it can connect again. A client that connects late gets
  the next export, not a replay.
- A stale socket file from an earlier run is removed at startup. A path that
  holds something other than a socket is an error, and nothing is deleted.
  The socket file is removed on a clean exit.
- An export error never stops the dashboard. The first error shows in the
  status line (`--export-json: ...` or `--export-sock: ...`). File writes
  are tried again on every later export; a file error shows once three
  writes in a row have failed, so one that clears by the next export (a
  reader holding the file open on Windows) is not reported.

## Fields

Unknown or absent values are `null`, never `0`. Floats that are not finite
are `null`. Floats are rounded to 2 decimals.

Top level:

| Field | Type | Meaning |
|-------|------|---------|
| `schema` | string | Always `"llm-visuals.stats/1"`. |
| `seq` | integer | Export counter, starts at 0 for each run. |
| `ts` | number | Unix time in seconds, with decimals. |
| `host` | string or null | Host name. |
| `source` | string | `"live"`, `"demo"` or `"replay"`. |
| `exporter_pid` | integer | Process ID of the dashboard. |
| `models` | array | One entry per monitored model. |
| `gpus` | array | One entry per GPU. |

`models[]`:

| Field | Type | Meaning |
|-------|------|---------|
| `pid`, `name`, `engine` | integer, string, string | Server process, model name, engine. |
| `quant` | string or null | Quantisation label. |
| `phase` | string | `"idle"`, `"prefill"` or `"decode"`. |
| `decode_tps`, `prefill_tps` | number or null | Current rates; the values `--log-db` writes. |
| `peak_decode_tps`, `peak_prefill_tps` | number or null | Peak rates of the session. |
| `ctx_used` | integer | Context tokens in use. |
| `ctx_max` | integer or null | Context size; `null` if unknown. |
| `session_decoded`, `session_prefilled` | integer | Token totals of the session. |
| `requests_finished` | integer | Requests finished since the dashboard started. |
| `gpu_indices` | array of integer | GPUs the model runs on; empty if unknown. |
| `current_request` | object or null | The request in progress: `prompt_tokens`, `cached_tokens`, `decoded`, `ttft_s`, `elapsed_s`. |
| `last_request` | object or null | The last finished request: `prompt_tokens`, `cached_tokens`, `decoded`, `ttft_s`, `duration_s`, `avg_prefill_tps`, `avg_decode_tps`, `peak_decode_tps`. Same values as the `requests` table of `--log-db`. |

`decode_tps` and `prefill_tps` are live rates over a sliding window of about
one second, so `decode_tps` ramps up over the first second of a request.
`prefill_tps` is a live rate only for engines that report prompt progress
while the prompt is read. Some engines (for example Maya, where
`live.prompt_read` is null) give it only when the prefill ends, so during
`phase: "prefill"` it can read 0. A consumer should then fall back to
`last_request.avg_prefill_tps`.

`ttft_s` is `null` until the first token is seen, or when it cannot be known.

`gpus[]`: `index`, `name`, `util_pct`, `mem_used_mb`, `mem_total_mb`,
`power_w`, `power_max_w` (`null` if unknown), `temp_c`, `mem_temp_c`,
`temp_limit_c`, `clock_sm_mhz`, `clock_mem_mhz` (`null` if unknown),
`fan_pct`, `fan_rpm`, `unified`.
When `unified` is `true` the memory fields describe system RAM, not a card.

## Added fields: CPU and the data path

These fields were added to `llm-visuals.stats/1` after its first release.
They are additive: a client that does not know them ignores them. A value
that is not known is `null`, never `0`.

### `system`

The machine as a whole, one object per export. It is present even with no
model attached.

| Field | Unit | Meaning |
|-------|------|---------|
| `cpu_pct` | % | Busy share of all logical CPUs since the last host sample (user, nice, system, irq, softirq, steal). Time waiting for I/O (`iowait`) is not busy. |
| `iowait_pct` | % | Share of CPU time spent idle with I/O in flight. A saturated SSD shows here. |
| `cpu_logical` | count | Logical CPUs (lines `cpuN` of `/proc/stat`). |
| `cpu_physical` | count | Physical cores: distinct (`physical id`, `core id`) pairs of `/proc/cpuinfo`, else of `/sys/devices/system/cpu/cpuN/topology`. |
| `ram_used_mb`, `ram_total_mb` | MiB | `MemTotal - MemAvailable`, and `MemTotal`. |

Sources: `/proc/stat` and `/proc/meminfo`, read every 0.4 s (`--poll-ms` if
longer). `cpu_pct` and `iowait_pct` are `null` until the second sample and
wherever there is no `/proc/stat` (macOS, Windows); `ram_*` come from the OS
on those systems. The figures are the host's, so llm-visuals must run on the
machine that runs the server.

### Per model: process CPU

| Field | Meaning |
|-------|---------|
| `label` | The engine's own display name for the model, when it gives one (Maya: `engine.model_label`, such as `DeepSeek V4.1 · EXL3 2.9 bpw`). Else `null`. |
| `process_cpu_cores` | CPU the server process used, in cores: the change of `utime + stime` in `/proc/<pid>/stat` over the host interval, divided by `CLK_TCK`. `CLK_TCK` is the constant 100 (the kernel's `USER_HZ`). `null` before the second sample and without `/proc`. |
| `engine_threads` | The threads the engine computes with. |
| `engine_threads_source` | Where that number comes from, first match wins: `"engine"` (the engine says: Maya `engine.cpu_threads`), `"cmdline"` (llama.cpp `-t N`, `--threads N` or `--threads=N`; not read for vLLM and SGLang), `"physical_cores"` (llama.cpp's own default), `"logical_cpus"`. |
| `process_cpu_pct` | `process_cpu_cores / engine_threads x 100`. It can pass 100 (a process may use more cores than it names, and a thread count can be a guess). `null` if either is `null`. |

### Per model: `bandwidth`

The weight path SSD -> RAM/CPU -> PCIe -> GPUs that the dashboard's
Bandwidth view draws. One object per model; the machine counters are the
same for every model, the weight streams are the model's.

| Field | Unit | Meaning |
|-------|------|---------|
| `disk_mb_s` | MB/s | Reads of every block device (`/proc/diskstats`). `null` without a sample or without `/proc`. |
| `proc_disk_mb_s` | MB/s | Reads the server process made from storage (`/proc/<pid>/io`). |
| `majflt_per_s` | 1/s | Major page faults of the process: weights paged in from disk. |
| `ram_gb_s` | GB/s (10^9 B) | Weights read out of system RAM by the CPU side, per second. `0` while idle where the stream is known. |
| `ram_source` | | `"estimated"`: the tensor table says how many bytes are CPU-side, times the step rate. `"maya"`: derived from Maya's expert tiers (below). `null`: unknown, and then `ram_gb_s` is `null` too. |
| `pcie` | | One entry per GPU with a PCIe counter (NVIDIA only; empty elsewhere): `gpu` (index), `rx_mb_s` (host to device), `tx_mb_s`, `cap_mb_s` (link capacity from its generation and width, `null` if unknown). |
| `vram` | | One entry per GPU: `gpu`, `busy_pct` (memory-controller busy; `null` on a unified-memory part, which has no such counter), `peak_gb_s` (datasheet memory bandwidth, `null` if unknown). |
| `vram_gb_s` | GB/s | Estimated weight stream out of VRAM across the cards (bytes per step x steps per second); `null` without a tensor table. |
| `vram_peak_gb_s` | GB/s | The ceiling of that stream for this model's split; `null` unless every card holding weights has a known peak. |
| `steps_per_s` | 1/s | Weight reads per second: decode steps (verification steps under speculative decoding), prefill batches while prefilling. |
| `bottleneck` | | The dashboard's verdict, below. |

**`bottleneck`.** The same rules as the Bandwidth view's verdict line.

| Field | Meaning |
|-------|---------|
| `stage` | `"disk"`, `"ram"`, `"pcie"`, `"vram"`, `"compute"` or `null` (no stage stands out). |
| `gpu` | The GPU the stage is about: for `"pcie"` the busiest link, for `"vram"` the busiest memory controller, for `"compute"` the busiest GPU. `null` for `"disk"`, `"ram"` and `null`. |
| `bound` | `true` when a real bound is named. `false` for an idle model, for the plain "prefill" label, and when nothing is saturated: do not draw those as a problem. |
| `headline`, `detail` | The verdict as text. |

Rules, in order, for decode: the disk is read heavily (system reads over
50 MB/s, the process over 20 MB/s, or over 50 major faults/s): `disk`; a
PCIe link over 35 % of its capacity: `pcie`; a memory controller over 75 %,
or the weight stream over 60 % of the cards' peak: `vram`; CPU-side weights
(over 2 % of the model) with every GPU under 60 % busy: `ram`; a GPU over
85 % busy: `compute`; else none. For prefill the PCIe threshold is 50 % and
a GPU over 60 % busy is `compute`. When idle only a disk being read counts.
`ram` needs a known tensor table, so an engine without one reports it only
where it can say so itself. Maya can: while a DeepSeek engine decodes, with
its CPU lane computing experts from RAM (`tiers.now.cpu` above 0), at least
85 % of `engine_threads` busy (`process_cpu_pct`) and no GPU over 85 % busy,
the stage is `ram` with `bound: true` (`gpu: null`).

### Maya: RAM stream from the expert tiers

Maya keeps most experts in system RAM. With no tensor table the dashboard
cannot estimate the stream, so for `engine: "maya"` it derives it from
`/metrics` `tiers.now`, while the model decodes:

```text
ram_gb_s = experts_per_token x bytes_per_expert x tok_s / 10^9

experts_per_token = tiers.now.cpu         (DeepSeek: experts the CPU lane computed from RAM)
                    else tiers.now.ram_fetch  (GLM: experts read from RAM to move to a card)
bytes_per_expert  = tiers.now.ram_gb x 2^30 / experts the pool holds
                    (tiers.now.ram_slots, else cpu_n, else ram_used)
tok_s             = tiers.now.tok_s       (the window the per-token counts cover)
```

`ram_gb` is in GiB: Maya's `expert_cache_mib` of 7180 MiB is `vram_gb` 7.0, and
the pool of 78.1 GiB over 11784 slots is 7.116 MB, the size of one VRAM slot
(`gpu_map` pool over `slots`). On a real DeepSeek V4.1 server (two RTX 3090,
14133 CPU experts, 181.4 GiB) the formula gave 19 to 22 GB/s over a 120-token answer,
steady while `tok_s` (10 to 23) and the per-token count (152 down to 63)
moved opposite ways, as a memory-bound stream does. It is an upper bound under speculative decoding,
where one verification step may read an expert once for several tokens. While
idle the figure is `0` (the tiers keep their last reading, nothing is
streaming); during prefill it is `null`. Maya's own `hardware` block (CPU, PCIe
and disk of its host) is not read: llm-visuals measures the same on the
host itself.

## Example

```json
{"schema":"llm-visuals.stats/1","seq":42,"ts":1767225600.25,"host":"box","source":"live","exporter_pid":4242,"models":[{"pid":1234,"name":"Qwen3-30B","engine":"llama.cpp","quant":"Q4_K_M","phase":"decode","decode_tps":71.4,"prefill_tps":0.0,"peak_decode_tps":80.2,"peak_prefill_tps":950.0,"ctx_used":2048,"ctx_max":32768,"session_decoded":9000,"session_prefilled":12000,"requests_finished":7,"gpu_indices":[0],"current_request":{"prompt_tokens":900,"cached_tokens":0,"decoded":120,"ttft_s":0.8,"elapsed_s":2.5},"last_request":null}],"gpus":[{"index":0,"name":"RTX 4090","util_pct":93.0,"mem_used_mb":20100,"mem_total_mb":24564,"power_w":310.5,"power_max_w":450.0,"temp_c":67.0,"mem_temp_c":null,"temp_limit_c":92.0,"clock_sm_mhz":2520,"clock_mem_mhz":10501,"fan_pct":55.0,"fan_rpm":null,"unified":false}]}
```

## Versioning

Inside `/1` only additive changes are made: new fields, and new values where
a field is documented as open. Fields are not removed, renamed or retyped.
Consumers must ignore fields they do not know. A breaking change gets a new
schema name, `llm-visuals.stats/2`.
