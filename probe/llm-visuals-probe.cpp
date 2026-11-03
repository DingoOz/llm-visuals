// llm-visuals-probe: one real forward pass through llama.cpp, reduced to the
// numbers the layer view (`llm-visuals --layers`) draws.
//
// It loads a GGUF with libllama, decodes a short prompt once, and watches the
// compute graph through the scheduler's eval callback (the hook llama.cpp's
// own `eval-callback` example uses). Only the residual stream is captured:
// the token embedding, the residual after each block's mixer (`ffn_inp`, or
// `attn_residual` on the qwen3.5 family), each block's output (`l_out`) and
// the final norm. Because the stream is captured whole, what each sublayer
// *wrote* is the difference between consecutive captures, which holds for
// any architecture whatever its sublayers are called.
//
// Output is JSON lines on stdout; llama.cpp's own log goes to stderr only
// with --verbose. Build with scripts/build-probe.sh against a llama.cpp
// checkout that has been built (it needs include/ and libllama).

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

namespace {

bool g_verbose = false;

void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

std::string json_str(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out + "\"";
}

// A float that is always valid JSON.
std::string num(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

// The tensors worth copying off the device, by their name before "-<layer>".
// Embedding names differ by architecture; the last one the graph produces
// before block 0 is the stream's true starting point.
const char * const WANTED[] = {
    "inp_embd", "inp_scaled", "model.input_embed", "model.embed_tokens",
    "ffn_inp", "attn_residual",
    "l_out",
    "result_norm",
};

bool split_name(const char * full, std::string & base, int & layer) {
    std::string s(full);
    layer = -1;
    auto dash = s.rfind('-');
    if (dash != std::string::npos && dash + 1 < s.size()) {
        bool digits = true;
        for (size_t i = dash + 1; i < s.size(); ++i) {
            digits = digits && s[i] >= '0' && s[i] <= '9';
        }
        if (digits) {
            layer = atoi(s.c_str() + dash + 1);
            s = s.substr(0, dash);
        }
    }
    base = s;
    for (const char * w : WANTED) {
        if (s == w) {
            return true;
        }
    }
    return false;
}

struct Capture {
    int64_t n_embd = 0;
    int64_t n_tokens = 0;
    // The last residual captured, [n_tokens][n_embd], for the deltas.
    std::vector<float> prev;
    int64_t prev_tokens = 0;
    std::vector<uint8_t> staging;
    std::vector<float> cur;
    int seq = 0;
};

float read_f(const uint8_t * data, ggml_type type, size_t off) {
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) (data + off);
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (data + off));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) (data + off));
        default:             return NAN;
    }
}

// Per-token L2 norm, mean |x| and max |x| over the embedding axis, as a
// JSON array of [l2, mean_abs, max_abs] triples.
std::string token_stats(const float * x, int64_t n_tok, int64_t n_embd) {
    std::string out = "[";
    for (int64_t t = 0; t < n_tok; ++t) {
        double sq = 0, sa = 0, mx = 0;
        const float * row = x + t * n_embd;
        for (int64_t i = 0; i < n_embd; ++i) {
            const double v = row[i];
            sq += v * v;
            sa += std::fabs(v);
            mx = std::max(mx, std::fabs(v));
        }
        if (t) {
            out += ",";
        }
        out += "[" + num(std::sqrt(sq)) + "," + num(sa / n_embd) + "," + num(mx) + "]";
    }
    return out + "]";
}

bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (Capture *) user_data;
    std::string base;
    int layer;
    const bool wanted = split_name(t->name, base, layer);
    if (ask) {
        return wanted;
    }
    if (!wanted || t->ne[2] != 1 || t->ne[3] != 1) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        return true;
    }
    const int64_t n_embd = t->ne[0];
    const int64_t n_tok  = t->ne[1];

    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        cap->staging.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, cap->staging.data(), 0, ggml_nbytes(t));
        data = cap->staging.data();
    }
    cap->cur.resize(n_embd * n_tok);
    for (int64_t j = 0; j < n_tok; ++j) {
        for (int64_t i = 0; i < n_embd; ++i) {
            cap->cur[j * n_embd + i] = read_f(data, t->type, j * t->nb[1] + i * t->nb[0]);
        }
    }

    std::string line = "{\"kind\":\"tensor\",\"seq\":" + std::to_string(cap->seq++) +
        ",\"name\":" + json_str(base) + ",\"layer\":" + std::to_string(layer) +
        ",\"n_embd\":" + std::to_string(n_embd) + ",\"n_tokens\":" + std::to_string(n_tok) +
        ",\"stream\":" + token_stats(cap->cur.data(), n_tok, n_embd);

    // What was added since the previous capture. A last block that only
    // keeps the output rows has fewer tokens; its rows are the last ones.
    const bool is_norm = base == "result_norm";
    if (!is_norm && !cap->prev.empty() && n_embd == cap->n_embd && n_tok <= cap->prev_tokens) {
        std::vector<float> delta(n_embd * n_tok);
        const int64_t skip = cap->prev_tokens - n_tok;
        for (int64_t j = 0; j < n_tok; ++j) {
            for (int64_t i = 0; i < n_embd; ++i) {
                delta[j * n_embd + i] = cap->cur[j * n_embd + i] - cap->prev[(j + skip) * n_embd + i];
            }
        }
        line += ",\"delta\":" + token_stats(delta.data(), n_tok, n_embd);
    }
    line += "}";
    puts(line.c_str());

    if (!is_norm) {
        cap->prev = cap->cur;
        cap->prev_tokens = n_tok;
        cap->n_embd = n_embd;
    }
    return true;
}

std::string piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return "";
    }
    return std::string(buf, n);
}

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf [-p PROMPT] [--ngl N] [--top K] [--verbose]\n"
        "  Decodes PROMPT once and prints per-layer residual-stream statistics\n"
        "  as JSON lines. --ngl -1 (default) offloads every layer.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox jumps over the lazy";
    int ngl = -1;
    int top_k = 5;
    const int max_tokens = 64;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage(argv[0]);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            model_path = next();
        } else if (a == "-p" || a == "--prompt") {
            prompt = next();
        } else if (a == "--ngl" || a == "-ngl") {
            ngl = atoi(next());
        } else if (a == "--top") {
            top_k = std::max(1, atoi(next()));
        } else if (a == "--verbose") {
            g_verbose = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (model_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(max_tokens + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), tokens.data(),
                           (int) tokens.size(), llama_vocab_get_add_bos(vocab), true);
    if (n < 0) {
        fprintf(stderr, "prompt is longer than %d tokens\n", max_tokens);
        return 1;
    }
    tokens.resize(std::min(n, max_tokens));
    if (tokens.empty()) {
        fprintf(stderr, "the prompt tokenized to nothing\n");
        return 1;
    }
    const int n_tok = (int) tokens.size();

    Capture cap;
    cap.n_tokens = n_tok;

    // One ubatch holds the whole prompt, so every captured tensor has every
    // token and the callback sees each layer exactly once.
    auto cparams = llama_context_default_params();
    cparams.n_ctx = std::max(n_tok, 64);
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = cparams.n_ctx;
    cparams.n_seq_max = 1;
    cparams.cb_eval = eval_cb;
    cparams.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create a context\n");
        return 1;
    }

    char desc[256] = {0};
    llama_model_desc(model, desc, sizeof(desc));
    char arch[64] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));

    std::string meta = "{\"kind\":\"meta\",\"arch\":" + json_str(arch) +
        ",\"desc\":" + json_str(desc) +
        ",\"n_layer\":" + std::to_string(llama_model_n_layer(model)) +
        ",\"n_embd\":" + std::to_string(llama_model_n_embd(model)) +
        ",\"ngl\":" + std::to_string(ngl) +
        ",\"prompt\":" + json_str(prompt) + ",\"tokens\":[";
    for (int i = 0; i < n_tok; ++i) {
        meta += std::string(i ? "," : "") + "{\"id\":" + std::to_string(tokens[i]) +
            ",\"piece\":" + json_str(piece(vocab, tokens[i])) + "}";
    }
    meta += "]}";
    puts(meta.c_str());
    fflush(stdout);

    // Every position asks for logits: the last block then keeps every row,
    // and each token gets its own next-token guess.
    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = n_tok;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n_tok; ++i) {
        const float * logits = llama_get_logits_ith(ctx, i);
        if (!logits) {
            continue;
        }
        float mx = -INFINITY;
        for (int v = 0; v < n_vocab; ++v) {
            mx = std::max(mx, logits[v]);
        }
        double z = 0;
        for (int v = 0; v < n_vocab; ++v) {
            z += std::exp((double) logits[v] - mx);
        }
        std::vector<int> idx(n_vocab);
        for (int v = 0; v < n_vocab; ++v) {
            idx[v] = v;
        }
        const int k = std::min(top_k, n_vocab);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        std::string line = "{\"kind\":\"next\",\"pos\":" + std::to_string(i) + ",\"top\":[";
        for (int j = 0; j < k; ++j) {
            const double p = std::exp((double) logits[idx[j]] - mx) / z;
            line += std::string(j ? "," : "") + "{\"id\":" + std::to_string(idx[j]) +
                ",\"piece\":" + json_str(piece(vocab, idx[j])) + ",\"p\":" + num(p) + "}";
        }
        line += "]}";
        puts(line.c_str());
    }
    puts("{\"kind\":\"done\"}");
    fflush(stdout);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
