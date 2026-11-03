# llm-visuals-probe

The forward-pass helper behind `llm-visuals --layers`. It links against your
own llama.cpp build, so it runs with whatever backends that build has (CUDA,
Metal, Vulkan, CPU) and supports every architecture llama.cpp does.

```sh
# llama.cpp built with CMake first: cmake -B build && cmake --build build -j
scripts/build-probe.sh ~/llama.cpp            # -> target/llm-visuals-probe
scripts/build-probe.sh ~/llama.cpp ~/.local/bin/llm-visuals-probe
```

`llm-visuals` looks for the probe in this order: `--probe PATH`,
`$LLM_VISUALS_PROBE`, beside its own executable, `target/` of this checkout,
then `PATH`.

## What it does

1. Loads the GGUF (`--ngl -1` offloads every layer) and tokenizes the prompt
   (64 tokens at most), with a context big enough to decode it in one
   micro-batch, so each tensor is seen once with every token.
2. Decodes it once, asking for logits at every position, so the last block
   keeps every row too.
3. Observes the graph through the scheduler's eval callback and copies off
   the device only the residual-stream tensors: the embedding (`inp_embd`,
   `inp_scaled`, `model.input_embed`, …), the mid-block residual (`ffn_inp`,
   or `attn_residual` on the qwen3.5 family), each block's output (`l_out`)
   and `result_norm`.
4. For each one it prints, per token, the L2 norm, mean |x| and max |x| of
   the stream, and of the difference from the previous capture: what that
   sublayer wrote.
5. Prints the top-5 next-token guesses (softmax probabilities) at each
   position.

## Output

JSON lines on stdout; llama.cpp's own log goes to stderr only with
`--verbose`.

```
{"kind":"meta","arch":"qwen35","desc":"…","n_layer":32,"n_embd":2560,"ngl":-1,"prompt":"…","tokens":[{"id":760,"piece":"The"},…]}
{"kind":"tensor","seq":1,"name":"attn_residual","layer":0,"n_embd":2560,"n_tokens":5,"stream":[[l2,mean_abs,max_abs],…],"delta":[[…],…]}
{"kind":"next","pos":4,"top":[{"id":11751,"piece":" Paris","p":0.524},…]}
{"kind":"done"}
```

A run without `done` is treated as failed rather than drawn partially.
