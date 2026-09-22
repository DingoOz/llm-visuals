## Summary

<!-- What changed, and why. Link the issue if there is one. -->

## Test plan

- [ ] `cargo test`
- [ ] `cargo build` has no warnings
- [ ] A parser or a calculation changed, and a unit test covers the new sample
- [ ] Layouts checked at 150×46, 100×30, and the `p` / `m` zoom views (`tmux capture-pane -e -p` is enough)
- [ ] Still builds for the six release targets. Anything that reads `/proc`, runs `nvidia-smi`, or hard-codes a path has a `cfg` branch and a non-Linux fallback

No API keys, bearer tokens, or private paths in the diff or the description.
