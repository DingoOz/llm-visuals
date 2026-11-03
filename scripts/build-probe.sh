#!/usr/bin/env bash
# Build llm-visuals-probe, the forward-pass helper behind `llm-visuals --layers`,
# against a llama.cpp checkout that has already been built with CMake.
#
#   scripts/build-probe.sh ~/llama.cpp            # -> target/llm-visuals-probe
#   scripts/build-probe.sh ~/llama.cpp ~/bin/llm-visuals-probe
#
# The probe links libllama dynamically and records the library directory as
# its rpath, so it runs with the same backends (CUDA, Metal, ...) llama.cpp
# was built with.
set -euo pipefail

llama=${1:?usage: $0 LLAMA_CPP_DIR [OUTPUT]}
out=${2:-"$(dirname "$0")/../target/llm-visuals-probe"}
src="$(dirname "$0")/../probe/llm-visuals-probe.cpp"
llama=$(cd "$llama" && pwd)

libdir=""
for d in "$llama/build/bin" "$llama/build/src" "$llama/build/lib" "$llama/build"; do
    if ls "$d"/libllama.* >/dev/null 2>&1; then
        libdir=$d
        break
    fi
done
if [ -z "$libdir" ]; then
    echo "no libllama under $llama/build; build llama.cpp first (cmake -B build && cmake --build build)" >&2
    exit 1
fi
# ggml may sit in its own directory in older trees.
ggmldir=$libdir
for d in "$llama/build/ggml/src" "$libdir"; do
    if ls "$d"/libggml.* >/dev/null 2>&1; then
        ggmldir=$d
        break
    fi
done

mkdir -p "$(dirname "$out")"
${CXX:-c++} -std=c++17 -O2 -Wall \
    -I"$llama/include" -I"$llama/ggml/include" \
    "$src" -o "$out" \
    -L"$libdir" -L"$ggmldir" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$libdir" -Wl,-rpath,"$ggmldir"
echo "built $out"
