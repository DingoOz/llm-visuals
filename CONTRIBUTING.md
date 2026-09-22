# Contributing

Thanks for looking. A few ground rules keep the dashboard honest and the code
easy to follow.

## Principles

1. **Show only real numbers.** Every value on screen must come from the server,
   the driver, or arithmetic on those. If something has to be a visual stand-in,
   it is labelled as such in the UI and in the README.
2. **Degrade, don't break.** A missing endpoint, an older server, a dense model,
   a 100×30 terminal or a 16-colour terminal must all render something useful.
3. **Smooth, don't flicker.** Rates go through `RateWindow`; visual levels go
   through `fade.rs`. Raw per-poll numbers are for the request log, not for
   meters.

## Workflow

```sh
cargo test                       # parsers, rate maths, layout helpers
cargo run -- --demo              # exercise every panel without hardware
cargo run                        # against a real server
```

The oldest toolchain that still builds is the `rust-version` in `Cargo.toml`
(currently 1.88). CI runs the tests on that compiler as well as on stable.

Before a PR:

- `cargo build` with no warnings.
- Add or update a unit test when touching a parser or a calculation.
- Check the three layouts: 150×46, 100×30 and the `p` / `m` zoom views.
  Capturing with `tmux capture-pane -e -p` and eyeballing is enough.
- Keep it building on all six released targets. CI compiles and tests on
  Linux, macOS and Windows and cross-compiles the ARM ones, but
  `./scripts/cross-build.sh` catches a break before you push — see
  [Building and cross-compiling](README.md#building-and-cross-compiling).
  Anything that reads `/proc`, runs `nvidia-smi` or hard-codes a path needs a
  `cfg` branch and a non-Linux fallback.

## Adding a metric

1. Parse it in `observe.rs` (or `gpu.rs`) with a unit test on a captured
   sample of the real payload.
2. Derive rates or windows in `perf.rs`; keep raw counters and derived values
   separate.
3. Render it in `render.rs`. Reuse `gauge`, `sparkline`, `panel` and
   `with_right`; add a `fmt_*` helper rather than formatting inline.
4. Feed the demo in `demo.rs` so `--demo` shows the new panel.
5. Document the source of the number in `README.md`.

## Adding a server

`model_detect.rs` recognises processes by name and command line and guesses the
port; `observe.rs` speaks llama.cpp's `/slots`. A new engine needs a detector
entry, a poller that fills `LiveStats`, and ideally a test payload.

## Style

Standard `rustfmt`. Comments explain *why*, not *what*. Keep functions under a
screen where possible; `render.rs` is long because each panel is one function,
which is the point.
