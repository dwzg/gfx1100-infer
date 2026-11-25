// gfxi: command-line front end.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <thread>
#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/chat.h"
#include "core/gguf.h"
#include "core/model.h"
#include "core/quant.h"
#include "core/ref.h"
#include "core/sampler.h"
#include "core/unicode.h"
#include "engine/gpu_model.h"
#include "engine/pack.h"
#include "vk/vk.h"
#include "core/tokenizer.h"

using namespace gfx;

static int cmd_info(int argc, char **argv) {
    if (argc < 1) { fprintf(stderr, "usage: gfxi info <model.gguf> [--tensors]\n"); return 2; }
    bool show_tensors = argc > 1 && !strcmp(argv[1], "--tensors");
    GGUF g; g.open(argv[0]);
    printf("file          %s (%.2f GB)\n", argv[0], g.file_size / 1e9);
    printf("architecture  %s\n", g.architecture.c_str());
    printf("name          %s\n", g.get_str("general.name").c_str());
    for (const auto &[k, v] : g.meta) {
        if (k.rfind(g.architecture + ".", 0) != 0) continue;
        if (v.type == M_STR) printf("  %-48s %.*s\n", k.c_str(), (int)v.s.size(), v.s.data());
        else if (v.type == M_ARR) {
            printf("  %-48s [%" PRIu64 "]", k.c_str(), v.count);
            if (v.elem_type != M_STR) for (uint64_t i = 0; i < v.count && i < 32; i++) printf(" %g", v.arr_float(i));
            printf("\n");
        } else if (v.type == M_F32 || v.type == M_F64) printf("  %-48s %g\n", k.c_str(), v.f);
        else printf("  %-48s %" PRId64 "\n", k.c_str(), v.i);
    }
    std::map<std::string, std::pair<uint64_t, uint64_t>> by_type;  // name -> (bytes, elems)
    for (const auto &t : g.tensors) { auto &e = by_type[type_info(t.type)->name]; e.first += t.nbytes; e.second += t.nelems; }
    printf("tensors       %zu\n", g.tensors.size());
    for (const auto &[name, e] : by_type)
        printf("  %-8s %8.3f GB  %8.2f M params\n", name.c_str(), e.first / 1e9, e.second / 1e6);
    for (const auto &t : g.tensors)
        if (!type_info(t.type)->decodable && t.type != T_I8 && t.type != T_I16 && t.type != T_I32 && t.type != T_I64 && t.type != T_F64)
            { printf("UNSUPPORTED weight encoding %s in tensor %s\n", type_info(t.type)->name, t.name.c_str()); break; }
    if (show_tensors)
        for (const auto &t : g.tensors) {
            std::string dims;
            for (uint64_t d : t.dims) dims += (dims.empty() ? "" : " x ") + std::to_string(d);
            printf("  %-52s %-8s [%s]\n", t.name.c_str(), type_info(t.type)->name, dims.c_str());
        }
    return 0;
}

// Memory plan: what the model needs on the GPU and how much context fits. Works on a header-only
// download too, since it only needs tensor shapes and types.
static int cmd_plan(int argc, char **argv) {
    if (argc < 1) { fprintf(stderr, "usage: gfxi plan <model.gguf> [--vram GiB] [--reserve GiB] [--ctx N]\n"); return 2; }
    double vram = 24.0, reserve = 0.75; long ctx = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--vram") vram = atof(next());
        else if (a == "--reserve") reserve = atof(next());
        else if (a == "--ctx") ctx = atol(next());
        else throw std::runtime_error("unknown option " + a);
    }
    GGUF g; g.open(argv[0], true);
    ModelDesc d = describe_model(g);
    const double GiB = 1073741824.0;
    // Weights that live on the GPU: every matrix the forward pass multiplies by, in its native layout
    // where one exists and as half precision otherwise. The embedding table stays on the CPU unless
    // it doubles as the output projection.
    uint64_t native = 0, fallback = 0, small = 0; std::map<std::string, uint64_t> fb_types;
    auto add = [&](const TensorInfo *t) {
        if (!t) return;
        if (t->dims.size() < 2 || t->dim(1) == 1) { small += t->nelems * 4; return; }
        const PackFormat *pf = pack_format(t->type);
        if (pf && t->dim(0) % pf->sb_cols == 0) native += (t->dim(0) / pf->sb_cols) * pf->sb_words * 4 * ((t->dim(1) + 63) / 64 * 64);
        else { fallback += t->nelems * 2; fb_types[type_info(t->type)->name] += t->nelems * 2; }
    };
    for (const LayerDesc &l : d.layers)
        for (const TensorInfo *t : {l.attn_norm, l.ffn_norm, l.wq, l.wk, l.wv, l.wo, l.bq, l.bk, l.bv, l.bo, l.q_norm, l.k_norm, l.wqkv, l.wz, l.conv, l.dt_bias,
                                    l.ssm_a, l.wbeta, l.walpha, l.ssm_norm, l.ssm_out, l.ffn_gate, l.ffn_up, l.ffn_down}) add(t);
    add(d.output_norm); add(d.output);
    uint64_t weights = native + fallback + small, rec = d.recurrent_state_bytes();
    uint64_t kv16 = GpuModel::bytes_per_token(d, false), kv8 = GpuModel::bytes_per_token(d, true);
    printf("model            %s (%s), %d layers, %d with a context-length cache\n", d.name.c_str(), d.arch_name.c_str(), d.n_layer, d.n_attn_layers());
    printf("weights on GPU   %.2f GiB  (native layouts %.2f, half-precision fallback %.2f)\n", weights / GiB, native / GiB, fallback / GiB);
    for (auto &[n, b] : fb_types) if (b > 0.005 * GiB) printf("                 fallback: %s tensors take %.2f GiB as half precision\n", n.c_str(), b / GiB);
    if (d.output != d.tok_embd) printf("embedding table  stays on the CPU (%.2f GiB in the file)\n", d.tok_embd->nbytes / GiB);
    printf("fixed state      %.2f GiB  (linear-attention layers, independent of context)\n", rec / GiB);
    printf("cache per token  %.1f KiB half precision, %.1f KiB 8-bit\n", kv16 / 1024.0, kv8 / 1024.0);
    double freeb = (vram - reserve) * GiB - (double)weights - (double)rec - 0.25 * GiB;  // 0.25 GiB working buffers
    printf("budget           %.1f GiB GPU memory, %.2f GiB kept free for the desktop and driver\n", vram, reserve);
    if (freeb <= 0) { printf("does not fit: weights alone exceed the budget by %.2f GiB\n", -freeb / GiB); return 1; }
    long max16 = (long)(freeb / kv16), max8 = (long)(freeb / kv8);
    auto capped = [&](long v) { return d.n_ctx_train > 0 && v > d.n_ctx_train ? d.n_ctx_train : v; };
    printf("context that fits  %ld tokens (half-precision cache), %ld tokens (8-bit cache); the model was trained for %d\n", capped(max16), capped(max8), d.n_ctx_train);
    if (d.n_ctx_train > 0 && max8 > d.n_ctx_train) printf("                   (memory alone would allow %ld / %ld; the trained limit caps it)\n", max16, max8);
    if (ctx > 0) printf("at %ld tokens      cache %.2f GiB half precision / %.2f GiB 8-bit -> %s\n", ctx, ctx * (double)kv16 / GiB, ctx * (double)kv8 / GiB,
                        ctx <= max16 ? "fits either way" : ctx <= max8 ? "fits with the 8-bit cache only" : "does not fit");
    // Bytes the GPU must read per generated token: all weights once, plus the cache once.
    double per_tok = (double)weights + (double)rec * 2;
    printf("memory read per token  %.2f GiB of weights and state, plus the filled cache (%.1f KiB per token of context)\n", per_tok / GiB, kv16 / 1024.0);
    for (long c : {4096L, 32768L, 131072L, 262144L})
        if (c <= max8) printf("  bandwidth ceiling at %6ld tokens of context: %5.1f tok/s (half-precision cache), %5.1f tok/s (8-bit), for 960 GB/s\n", c,
                              960e9 / (per_tok + (double)c * kv16), 960e9 / (per_tok + (double)c * kv8));
    return 0;
}

// Decode one tensor to raw little-endian float32 (used by the test-suite to compare decoders).
static int cmd_dequant(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: gfxi dequant <model.gguf> <tensor> <out.f32> [max_elems]\n"); return 2; }
    GGUF g; g.open(argv[0]);
    const TensorInfo *t = g.tensor(argv[1]);
    if (!t) { fprintf(stderr, "no tensor named %s\n", argv[1]); return 1; }
    const TypeInfo *ti = type_info(t->type);
    uint64_t n = t->nelems;
    if (argc > 3) { uint64_t mx = strtoull(argv[3], nullptr, 10); mx -= mx % ti->block_elems; if (mx < n) n = mx; }
    std::vector<float> out(n);
    if (!dequantize(t->type, t->data, out.data(), n)) { fprintf(stderr, "no decoder for %s\n", ti->name); return 1; }
    FILE *f = fopen(argv[2], "wb");
    if (!f || fwrite(out.data(), 4, n, f) != n) { perror(argv[2]); return 1; }
    fclose(f);
    return 0;
}

// Decode raw block data of a given type id (test helper for encodings without a small test model).
static int cmd_dequant_raw(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: gfxi dequant-raw <type-id> <in.bin> <out.f32>\n"); return 2; }
    uint32_t type = (uint32_t)atoi(argv[0]);
    const TypeInfo *ti = type_info(type);
    if (!ti) { fprintf(stderr, "unknown type id\n"); return 1; }
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    std::vector<uint8_t> in; uint8_t buf[65536]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) in.insert(in.end(), buf, buf + r);
    fclose(f);
    uint64_t nb = in.size() / ti->block_bytes, n = nb * ti->block_elems;
    std::vector<float> out(n);
    if (!dequantize(type, in.data(), out.data(), n)) { fprintf(stderr, "no decoder for %s\n", ti->name); return 1; }
    f = fopen(argv[2], "wb");
    if (!f || fwrite(out.data(), 4, n, f) != n) { perror(argv[2]); return 1; }
    fclose(f);
    return 0;
}

static double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

static std::string read_arg_text(const char *arg) {
    if (arg[0] != '@') return arg;
    FILE *f = fopen(arg + 1, "rb"); if (!f) throw std::runtime_error(std::string("cannot open ") + (arg + 1));
    std::string s; char buf[65536]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, r);
    fclose(f); return s;
}

static int cmd_tokenize(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gfxi tokenize <model.gguf> <text | @file> [--no-special]\n"); return 2; }
    GGUF g; g.open(argv[0]);
    Tokenizer tok(g);
    bool special = !(argc > 2 && !strcmp(argv[2], "--no-special"));
    std::vector<int32_t> ids = tok.encode(read_arg_text(argv[1]), false, special);
    for (size_t i = 0; i < ids.size(); i++) printf("%s%d", i ? " " : "", ids[i]);
    printf("\n");
    return 0;
}

static int cmd_run(int argc, char **argv) {
    if (argc < 1) {
        fprintf(stderr, "usage: gfxi run <model.gguf> -p <prompt | @file> [-n tokens] [--chat [--system S] [--no-think]] [--temp T] [--top-k K] [--top-p P]\n"
                        "                [--seed S] [--ctx N|max] [--reserve GiB] [--threads N] [--device gpu|cpu|<name>] [--kv8] [--batch N] [--logits out.f32]\n"
                        "                [--save-state file] [--load-state file]   keep / reuse a processed prompt\n");
        return 2;
    }
    std::string prompt, logits_path, save_path, load_path, system_prompt, device = "gpu"; bool think = true; int n_predict = 64, n_ctx = 0, threads = (int)std::thread::hardware_concurrency(); bool chat = false, kv8 = false, ctx_max = false; double reserve_gib = 0.25; int batch = 32;
    SamplerParams sp; sp.temperature = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "-p") prompt = read_arg_text(next());
        else if (a == "-n") n_predict = atoi(next());
        else if (a == "--chat") chat = true;
        else if (a == "--system") system_prompt = read_arg_text(next());
        else if (a == "--no-think") think = false;
        else if (a == "--temp") sp.temperature = (float)atof(next());
        else if (a == "--top-k") sp.top_k = atoi(next());
        else if (a == "--top-p") sp.top_p = (float)atof(next());
        else if (a == "--seed") sp.seed = strtoull(next(), nullptr, 10);
        else if (a == "--ctx") { std::string v = next(); if (v == "max") ctx_max = true; else n_ctx = atoi(v.c_str()); }
        else if (a == "--reserve") reserve_gib = atof(next());
        else if (a == "--threads") threads = atoi(next());
        else if (a == "--logits") logits_path = next();
        else if (a == "--device") device = next();
        else if (a == "--kv8") kv8 = true;
        else if (a == "--batch") batch = atoi(next());
        else if (a == "--save-state") save_path = next();
        else if (a == "--load-state") load_path = next();
        else throw std::runtime_error("unknown option " + a);
    }
    if ((!save_path.empty() || !load_path.empty()) && device == "cpu") throw std::runtime_error("saved contexts need the GPU path");
    GGUF g; g.open(argv[0]);
    ModelDesc desc = describe_model(g);
    Tokenizer tok(g);
    if (chat) {
        ChatFormat cf = detect_chat_format(g);
        if (!cf.supported) throw std::runtime_error("this model's chat template (" + cf.name + ") is not supported; pass the full prompt with -p instead");
        prompt = (system_prompt.empty() ? "" : cf.message("system", system_prompt)) + cf.message("user", prompt) + cf.reply_start(think);
    }
    // When continuing a saved context the prompt is appended to it, so no start-of-text token.
    std::vector<int32_t> ids = tok.encode(prompt, load_path.empty(), true);
    if (ids.empty() && load_path.empty()) throw std::runtime_error("the prompt is empty");
    if (n_ctx <= 0 && !load_path.empty()) ctx_max = true;
    if (n_ctx <= 0) n_ctx = (int)ids.size() + n_predict + 1;
    std::unique_ptr<RefModel> ref; std::unique_ptr<Device> dev; std::unique_ptr<GpuModel> gpu;
    std::function<const std::vector<float> &(const int32_t *, int)> eval;
    if (device == "cpu") {
        fprintf(stderr, "[reference CPU path] %s: %d layers (%d full attention), %d prompt tokens\n", desc.name.c_str(), desc.n_layer, desc.n_attn_layers(), (int)ids.size());
        ref = std::make_unique<RefModel>(g, desc, n_ctx, threads);
        eval = [&](const int32_t *t, int n) -> const std::vector<float> & { return ref->eval(t, n); };
    } else {
        dev = std::make_unique<Device>(device == "gpu" ? "" : device);
        fprintf(stderr, "[%s] loading %s ...\n", dev->caps().name.c_str(), desc.name.c_str());
        gpu = std::make_unique<GpuModel>(*dev, g, desc, ctx_max ? 0 : n_ctx, false, kv8, (uint64_t)(reserve_gib * 1073741824.0), batch);
        if (dev->caps().coop_matrix) fprintf(stderr, "[matrix instructions verified for: %s]\n", gpu->wmma_report().empty() ? "(disabled)" : gpu->wmma_report().c_str());
        if ((int)ids.size() >= gpu->n_ctx()) throw std::runtime_error("the prompt (" + std::to_string(ids.size()) + " tokens) does not fit the context (" + std::to_string(gpu->n_ctx()) + ")");
        fprintf(stderr, "[%s] weights %.2f GiB, context %d tokens (%s cache, %.2f GiB), %zu dispatches and %zu sync points per token, %d prompt tokens\n", dev->caps().name.c_str(),
                gpu->weight_bytes() / 1073741824.0, gpu->n_ctx(), kv8 ? "8-bit" : "half-precision", gpu->state_bytes() / 1073741824.0, gpu->dispatches_per_token(), gpu->barriers_per_token(), (int)ids.size());
        eval = [&](const int32_t *t, int n) -> const std::vector<float> & { return gpu->eval(t, n); };
    }
    Sampler sampler(sp);
    std::vector<int32_t> history;
    if (!load_path.empty()) {
        auto tl = std::chrono::steady_clock::now();
        history = gpu->load_state(load_path);
        fprintf(stderr, "[restored %d tokens of context from %s in %.1f s]\n", gpu->n_past(), load_path.c_str(), seconds_since(tl));
        if (gpu->n_past() + (int)ids.size() >= gpu->n_ctx()) throw std::runtime_error("the saved context plus the prompt does not fit the context");
    }
    auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> *logits = ids.empty() ? &gpu->logits() : &eval(ids.data(), (int)ids.size());
    auto t1 = std::chrono::steady_clock::now();
    if (!save_path.empty()) {
        history.insert(history.end(), ids.begin(), ids.end());
        gpu->save_state(save_path, history);
        fprintf(stderr, "[saved %d tokens of context to %s]\n", gpu->n_past(), save_path.c_str());
    }
    if (!logits_path.empty()) {
        FILE *f = fopen(logits_path.c_str(), "wb");
        if (!f || fwrite(logits->data(), 4, logits->size(), f) != logits->size()) throw std::runtime_error("cannot write " + logits_path);
        fclose(f);
    }
    std::string pending; int n_gen = 0;
    for (; n_gen < n_predict; n_gen++) {
        int32_t t = sampler.sample(logits->data(), (int)logits->size());
        if (tok.is_eog(t)) break;
        pending += tok.piece(t);
        size_t ok = utf8_complete_prefix(pending);
        fwrite(pending.data(), 1, ok, stdout); fflush(stdout); pending.erase(0, ok);
        if (n_gen + 1 < n_predict) logits = &eval(&t, 1);
    }
    fwrite(pending.data(), 1, pending.size(), stdout); printf("\n");
    auto t2 = std::chrono::steady_clock::now();
    double pre = std::chrono::duration<double>(t1 - t0).count(), gen = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "[prompt %d tokens in %.2f s, generated %d tokens in %.2f s (%.2f tok/s)]\n", (int)ids.size(), pre, n_gen, gen, n_gen > 1 ? (n_gen - 1) / gen : 0.0);
    return 0;
}

// Interactive chat: one line of input per turn, the context is kept between turns.
static int cmd_chat(int argc, char **argv) {
    if (argc < 1) {
        fprintf(stderr, "usage: gfxi chat <model.gguf> [--system S] [--no-think] [-n max reply tokens] [--ctx N|max] [--reserve GiB] [--kv8]\n"
                        "                 [--temp T] [--top-k K] [--top-p P] [--seed S] [--batch N] [--device name]\n"
                        "                 [--load-state file]   start from a saved context (for example a long document)\n");
        return 2;
    }
    GGUF g; g.open(argv[0]);
    ModelDesc desc = describe_model(g);
    Tokenizer tok(g);
    ChatFormat cf = detect_chat_format(g);
    if (!cf.supported) throw std::runtime_error("this model's chat template (" + cf.name + ") is not supported");
    std::string system_prompt, device, load_path; bool think = true, kv8 = false, ctx_max = false; int n_reply = 2048, n_ctx = 8192, batch = 32; double reserve_gib = 0.25;
    // Sampling defaults: what the model file recommends, if it says.
    SamplerParams sp;
    sp.temperature = (float)g.get_float("general.sampling.temp", 0.7); sp.top_k = (int)g.get_int("general.sampling.top_k", 20); sp.top_p = (float)g.get_float("general.sampling.top_p", 0.8);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--system") system_prompt = read_arg_text(next());
        else if (a == "--no-think") think = false;
        else if (a == "-n") n_reply = atoi(next());
        else if (a == "--ctx") { std::string v = next(); if (v == "max") ctx_max = true; else n_ctx = atoi(v.c_str()); }
        else if (a == "--reserve") reserve_gib = atof(next());
        else if (a == "--kv8") kv8 = true;
        else if (a == "--temp") sp.temperature = (float)atof(next());
        else if (a == "--top-k") sp.top_k = atoi(next());
        else if (a == "--top-p") sp.top_p = (float)atof(next());
        else if (a == "--seed") sp.seed = strtoull(next(), nullptr, 10);
        else if (a == "--batch") batch = atoi(next());
        else if (a == "--device") device = next();
        else if (a == "--load-state") { load_path = next(); ctx_max = true; }
        else throw std::runtime_error("unknown option " + a);
    }
    Device dev(device);
    fprintf(stderr, "[%s] loading %s ...\n", dev.caps().name.c_str(), desc.name.c_str());
    GpuModel gpu(dev, g, desc, ctx_max ? 0 : n_ctx, false, kv8, (uint64_t)(reserve_gib * 1073741824.0), batch);
    if (dev.caps().coop_matrix) fprintf(stderr, "[matrix instructions verified for: %s]\n", gpu.wmma_report().empty() ? "(disabled)" : gpu.wmma_report().c_str());
    Sampler sampler(sp);
    bool first = true;
    if (!load_path.empty()) { gpu.load_state(load_path); first = false; fprintf(stderr, "[restored %d tokens of context]\n", gpu.n_past()); }
    fprintf(stderr, "[context %d tokens, %s cache; temperature %.2f, top-k %d, top-p %.2f; end input (Ctrl-D) to quit]\n", gpu.n_ctx(), kv8 ? "8-bit" : "half-precision",
            sp.temperature, sp.top_k, sp.top_p);
    bool reply_open = false;   // the previous reply still needs its closing marker
    std::string line;
    for (;;) {
        fprintf(stderr, "\n> "); fflush(stderr);
        line.clear();
        int ch; while ((ch = getchar()) != EOF && ch != '\n') line += (char)ch;
        if (ch == EOF && line.empty()) break;
        if (line.empty()) continue;
        std::string text = reply_open ? cf.reply_end() : "";
        if (first && !system_prompt.empty()) text += cf.message("system", system_prompt);
        text += cf.message("user", line) + cf.reply_start(think);
        std::vector<int32_t> ids = tok.encode(text, first, true);
        first = false;
        if (gpu.n_past() + (int)ids.size() + 1 >= gpu.n_ctx()) { fprintf(stderr, "[the context is full: %d of %d tokens used]\n", gpu.n_past(), gpu.n_ctx()); break; }
        auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> *logits = &gpu.eval(ids.data(), (int)ids.size());
        double t_prompt = seconds_since(t0);
        std::string pending; int n_gen = 0;
        auto t1 = std::chrono::steady_clock::now();
        while (n_gen < n_reply && gpu.n_past() + 1 < gpu.n_ctx()) {
            int32_t t = sampler.sample(logits->data(), (int)logits->size());
            if (tok.is_eog(t)) break;
            pending += tok.piece(t);
            size_t ok = utf8_complete_prefix(pending);
            fwrite(pending.data(), 1, ok, stdout); fflush(stdout); pending.erase(0, ok);
            logits = &gpu.eval(&t, 1); n_gen++;
        }
        fwrite(pending.data(), 1, pending.size(), stdout); printf("\n"); fflush(stdout);
        reply_open = true;
        fprintf(stderr, "[%d prompt tokens in %.2f s, %d reply tokens at %.1f tok/s, context %d / %d]\n", (int)ids.size(), t_prompt, n_gen,
                n_gen ? n_gen / seconds_since(t1) : 0.0, gpu.n_past(), gpu.n_ctx());
    }
    return 0;
}

// Run the GPU engine and the CPU reference on the same tokens and report how closely they agree.
static int cmd_compare(int argc, char **argv) {
    if (argc < 1) { fprintf(stderr, "usage: gfxi compare <model.gguf> [-p prompt] [-n extra tokens] [--device name] [--tol T] [--kv8] [--skip N]\n"); return 2; }
    std::string prompt = "The quick brown fox jumps over the lazy dog. In 1969, humans first", device; int n_extra = 8; double tol = 0.05; bool fp16 = false, kv8 = false; int skip = 0, batch = 32;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "-p") prompt = read_arg_text(next());
        else if (a == "-n") n_extra = atoi(next());
        else if (a == "--device") device = next();
        else if (a == "--tol") tol = atof(next());
        else if (a == "--fp16") fp16 = true;
        else if (a == "--kv8") kv8 = true;
        else if (a == "--skip") skip = atoi(next());   // start at this position, over an all-zero cache
        else if (a == "--batch") batch = atoi(next());
        else throw std::runtime_error("unknown option " + a);
    }
    GGUF g; g.open(argv[0]);
    ModelDesc desc = describe_model(g);
    Tokenizer tok(g);
    std::vector<int32_t> ids = tok.encode(prompt, true, true);
    int n_ctx = skip + (int)ids.size() + n_extra + 1;
    Device dev(device);
    GpuModel gpu(dev, g, desc, n_ctx, fp16, kv8, 256ull << 20, batch);
    printf("weights on device: %.3f GiB\n", gpu.weight_bytes() / 1073741824.0);
    if (!gpu.wmma_report().empty()) printf("matrix-instruction kernels verified for: %s\n", gpu.wmma_report().c_str());
    RefModel ref(g, desc, n_ctx, (int)std::thread::hardware_concurrency());
    if (skip > 0) { gpu.skip_to(skip); ref.skip_to(skip); }
    printf("device %s, model %s, %zu prompt tokens + %d generated, starting at position %d\n", dev.caps().name.c_str(), desc.name.c_str(), ids.size(), n_extra, skip);
    double worst = 0; int mismatches = 0, steps = 0;
    auto step = [&](const int32_t *t, int n) -> int32_t {
        const std::vector<float> &lr = ref.eval(t, n), &lg = gpu.eval(t, n);
        double num = 0, den = 0, maxd = 0;
        for (size_t i = 0; i < lr.size(); i++) { double d = (double)lg[i] - lr[i]; num += d * d; den += (double)lr[i] * lr[i]; maxd = std::max(maxd, std::fabs(d)); }
        int32_t ar = (int32_t)(std::max_element(lr.begin(), lr.end()) - lr.begin()), ag = (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        double rel = std::sqrt(num / std::max(den, 1e-30));
        printf("  step %2d: relative error %.2e, max abs diff %.4f, top token %s\n", steps, rel, maxd, ar == ag ? "same" : "DIFFERENT");
        worst = std::max(worst, rel); mismatches += ar != ag; steps++;
        return ar;
    };
    int32_t t = step(ids.data(), (int)ids.size());
    for (int i = 0; i < n_extra; i++) t = step(&t, 1);
    bool ok = worst <= tol && mismatches == 0;
    printf("%s: worst relative error %.2e (tolerance %.2e), %d top-token mismatches in %d steps\n", ok ? "PASS" : "FAIL", worst, tol, mismatches, steps);
    return ok ? 0 : 1;
}

static int cmd_devices() {
    auto names = Device::list_devices();
    if (names.empty()) { printf("no Vulkan devices found\n"); return 1; }
    for (size_t i = 0; i < names.size(); i++) printf("%zu: %s\n", i, names[i].c_str());
    Device dev;
    const DeviceCaps &c = dev.caps();
    printf("\nselected: %s (driver %s)\n", c.name.c_str(), c.driver.c_str());
    printf("  GPU memory            %.1f GiB, %.1f GiB free right now\n", c.vram_bytes / 1073741824.0, dev.vram_free() / 1073741824.0);
    printf("  wave size             %u (selectable %u..%u: %s)\n", c.subgroup_size, c.min_subgroup, c.max_subgroup, c.subgroup_size_control ? "yes" : "no");
    printf("  matrix instructions   %s\n", c.coop_matrix ? "yes (VK_KHR_cooperative_matrix)" : "no");
    printf("  packed integer dot    %s\n", c.int_dot_4x8_mixed_accel ? "yes, hardware accelerated" : c.int_dot ? "supported, not accelerated" : "no");
    printf("  largest buffer bind   %.2f GiB\n", c.max_storage_range / 1073741824.0);
    return 0;
}

// Device benchmark: achievable memory bandwidth, the throughput of each quantised matvec kernel
// relative to it, and the fixed cost of a dispatch. With a model: tokens per second.
static int cmd_bench(int argc, char **argv) {
    std::string device, model; uint64_t mib = 512; uint32_t wave = 0; int n_tokens = 64, batch = 32; bool kv8 = false, no_wmma = false;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--device") device = next();
        else if (a == "--mib") mib = strtoull(next(), nullptr, 10);
        else if (a == "--wave") wave = (uint32_t)atoi(next());
        else if (a == "-n") n_tokens = atoi(next());
        else if (a == "--kv8") kv8 = true;
        else if (a == "--batch") batch = atoi(next());
        else if (a == "--no-wmma") no_wmma = true;
        else model = a;
    }
    Device dev(device);
    const DeviceCaps &c = dev.caps();
    printf("device %s (driver %s), kernels built for wave size %s\n", c.name.c_str(), c.driver.c_str(), wave ? std::to_string(wave).c_str() : "default");
    if (!model.empty()) {
        GGUF g; g.open(model);
        ModelDesc desc = describe_model(g);
        int n_ctx = n_tokens + 8;
        auto t0 = std::chrono::steady_clock::now();
        GpuModel gpu(dev, g, desc, n_ctx, false, kv8, 256ull << 20, batch, !no_wmma);
        printf("%s: loaded in %.1f s, weights %.2f GiB, %zu dispatches and %zu sync points per token\n", desc.name.c_str(), seconds_since(t0),
               gpu.weight_bytes() / 1073741824.0, gpu.dispatches_per_token(), gpu.barriers_per_token());
        if (dev.caps().coop_matrix) printf("matrix-instruction kernels verified for: %s\n", gpu.wmma_report().empty() ? "(disabled)" : gpu.wmma_report().c_str());
        std::vector<int32_t> toks(n_tokens);
        for (int i = 0; i < n_tokens; i++) toks[i] = (int32_t)((1000003ull * (i + 1)) % (uint64_t)desc.n_vocab);
        // Prompt speed: whole batches.
        int np = std::max(1, n_tokens / gpu.batch()) * gpu.batch();
        if (np <= n_tokens && gpu.batch() > 1) {
            gpu.eval(toks.data(), gpu.batch()); gpu.reset();   // warm-up: builds the batch program
            t0 = std::chrono::steady_clock::now();
            gpu.eval(toks.data(), np);
            double dt = seconds_since(t0);
            printf("prompt speed:     %.1f tok/s  (%d tokens in batches of %d)\n", np / dt, np, gpu.batch());
            gpu.reset();
        }
        // Generation speed: one token per submission.
        int ng = std::min(n_tokens, 34);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 2; i++) gpu.eval(toks.data() + i, 1);   // warm-up
        t0 = std::chrono::steady_clock::now();
        for (int i = 2; i < ng; i++) gpu.eval(toks.data() + i, 1);
        double dt = seconds_since(t0) / (ng - 2);
        printf("generation speed: %.2f ms per token = %.1f tok/s  (reads %.2f GiB per token -> %.0f GB/s effective)\n", dt * 1e3, 1.0 / dt,
               gpu.weight_bytes() / 1073741824.0, gpu.weight_bytes() / dt / 1e9);
        return 0;
    }
    const uint64_t limit = std::min<uint64_t>(c.max_storage_range, mib << 20);
    const uint32_t stride = 16384;
    Buffer y = dev.alloc(stride * 4ull), dummy = dev.alloc(256); dev.fill_zero(dummy);
    // 1. dispatch + sync overhead
    {
        Buffer xs = dev.alloc(256), x = dev.alloc(4096); dev.fill_zero(x);
        Program p(dev); uint32_t nb = 16; const int reps = 2000;
        for (int i = 0; i < reps; i++) p.dispatch(dev.pipeline("blocksum"), {R(x), W(xs)}, &nb, sizeof nb, 1);   // each rewrites xs -> one sync point each
        p.finish(); p.run();
        auto t0 = std::chrono::steady_clock::now(); p.run();
        printf("dispatch + sync point: %.2f us each\n", seconds_since(t0) / reps * 1e6);
    }
    // 2. bandwidth roofline
    uint32_t planes = (uint32_t)(limit / (stride * 4ull));
    uint64_t bytes = (uint64_t)planes * stride * 4;
    Buffer big = dev.alloc(bytes);
    {
        std::vector<uint32_t> rnd(1 << 22); uint32_t s = 99;
        for (auto &v : rnd) { s = s * 1664525u + 1013904223u; v = s; }
        for (uint64_t off = 0; off < bytes; off += rnd.size() * 4) dev.upload(big.sub(off, std::min<uint64_t>(rnd.size() * 4, bytes - off)), rnd.data(), std::min<uint64_t>(rnd.size() * 4, bytes - off));
    }
    double roof = 0;
    {
        struct { uint32_t n_planes, stride; } pc{planes, stride};
        Program p(dev); const int reps = 8;
        for (int i = 0; i < reps; i++) p.dispatch(dev.pipeline("memread", {}, wave), {R(big), W(y)}, &pc, sizeof pc, stride / 64);
        p.finish(); p.run();
        auto t0 = std::chrono::steady_clock::now(); p.run();
        roof = bytes * (double)reps / seconds_since(t0) / 1e9;
        printf("memory read roofline: %.0f GB/s  (%llu MiB, %u rows)\n", roof, (unsigned long long)(bytes >> 20), stride);
    }
    // 3. each matvec kernel on the same data, as a matrix with `stride` rows
    Buffer x = dev.alloc(4ull << 20), xs = dev.alloc(1 << 20); dev.fill_zero(x); dev.fill_zero(xs);
    for (uint32_t type : {(uint32_t)T_Q4_K, (uint32_t)T_Q5_K, (uint32_t)T_Q6_K, (uint32_t)T_Q3_K, (uint32_t)T_IQ4_XS, (uint32_t)T_IQ4_NL, (uint32_t)T_Q8_0, (uint32_t)T_NVFP4}) {
        const PackFormat *pf = pack_format(type);
        uint32_t n_sb = std::min<uint32_t>(planes / pf->sb_words, (4u << 20) / 4 / pf->sb_cols);
        struct { uint32_t n_sb, stride, n_rows, row0, flags; } pc{n_sb, stride, stride, 0, 0};
        uint64_t used = (uint64_t)n_sb * pf->sb_words * stride * 4;
        Program p(dev); const int reps = 4;
        for (int i = 0; i < reps; i++) p.dispatch(dev.pipeline(pf->kernel, {}, wave), {R(big), R(x), R(xs), W(y), R(dummy)}, &pc, sizeof pc, stride / 64);
        p.finish(); p.run();
        auto t0 = std::chrono::steady_clock::now(); p.run();
        double gbs = used * (double)reps / seconds_since(t0) / 1e9;
        printf("  %-7s %6.0f GB/s  (%3.0f%% of roofline, %.1f billion weights/s)\n", type_info(type)->name, gbs, 100.0 * gbs / roof,
               gbs * 8.0 / type_info(type)->bpw());
    }
    return 0;
}

// Check the kernels of one weight format against the scalar decoder, on raw block data of the
// given type laid out as a matrix with n_in columns and random input vectors: the single-vector
// kernel, the batched kernel, and the matrix-instruction kernel (emulated, and real where the
// device has the instructions).
static int cmd_test_matvec(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: gfxi test-matvec <type-id> <blocks.bin> <n_in> [--device name]\n"); return 2; }
    uint32_t type = (uint32_t)atoi(argv[0]), n_in = (uint32_t)atoi(argv[2]);
    std::string device = argc > 4 && !strcmp(argv[3], "--device") ? argv[4] : "";
    const TypeInfo *ti = type_info(type); const PackFormat *pf = pack_format(type);
    if (!ti || !pf) { fprintf(stderr, "type has no native GPU layout\n"); return 1; }
    if (n_in % pf->sb_cols) { fprintf(stderr, "n_in must be a multiple of %u\n", pf->sb_cols); return 1; }
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    std::vector<uint8_t> in; uint8_t buf[65536]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) in.insert(in.end(), buf, buf + r);
    fclose(f);
    const uint32_t NT = 16;   // tokens
    uint64_t row_bytes = (uint64_t)n_in / ti->block_elems * ti->block_bytes;
    uint32_t rows = (uint32_t)(in.size() / row_bytes) / 16 * 16, stride = (rows + 63) / 64 * 64;
    if (rows == 0) { fprintf(stderr, "not enough data for 16 rows\n"); return 1; }
    std::vector<float> x((size_t)NT * n_in), xs((size_t)NT * n_in / 16), want((size_t)NT * rows), wrow(n_in), got((size_t)NT * rows);
    std::vector<uint16_t> xh(x.size());
    uint32_t seed = 12345;
    for (auto &v : x) { seed = seed * 1664525u + 1013904223u; v = ((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
    for (size_t b = 0; b < xs.size(); b++) { float t = 0; for (int i = 0; i < 16; i++) t += x[16 * b + i]; xs[b] = t; }
    for (size_t i = 0; i < x.size(); i++) xh[i] = fp32_to_fp16(x[i]);
    double ref_norm = 0;
    for (uint32_t row = 0; row < rows; row++) {
        dequantize(type, in.data() + row * row_bytes, wrow.data(), n_in);
        for (uint32_t t = 0; t < NT; t++) {
            double acc = 0; const float *xt = &x[(size_t)t * n_in];
            for (uint32_t i = 0; i < n_in; i++) acc += (double)wrow[i] * xt[i];
            want[(size_t)t * rows + row] = (float)acc; ref_norm += acc * acc;
        }
    }
    Device dev(device);
    uint64_t words = (uint64_t)(n_in / pf->sb_cols) * pf->sb_words * stride;
    std::vector<uint32_t> packed(words);
    pack_rows(*pf, in.data(), n_in, 0, rows, stride, packed.data());
    Buffer bw = dev.alloc(words * 4), bx = dev.alloc(x.size() * 4), bxs = dev.alloc(xs.size() * 4), bxh = dev.alloc(xh.size() * 2), by = dev.alloc(got.size() * 4), dummy = dev.alloc(256);
    dev.upload(bw, packed.data(), words * 4); dev.upload(bx, x.data(), x.size() * 4); dev.upload(bxs, xs.data(), xs.size() * 4); dev.upload(bxh, xh.data(), xh.size() * 2); dev.fill_zero(dummy);
    Buffer bt;
    if (pf->needs_table) { uint32_t count; const uint32_t *data = pack_table(*pf, &count); bt = dev.alloc(count * 4ull); dev.upload(bt, data, count * 4ull); }
    struct PcBatch { uint32_t n_sb, stride, n_rows, row0, flags, n_tokens, x_stride, xs_stride, y_stride; } pb{n_in / pf->sb_cols, stride, rows, 0, 0, NT, n_in, n_in / 16, rows};
    struct { uint32_t n_sb, stride, n_rows, row0, flags; } p1{n_in / pf->sb_cols, stride, rows, 0, 0};
    std::string base = pf->kernel + 6;   // "matvec_x" -> "_x"
    int failed = 0;
    auto check = [&](const char *what, const std::string &kernel, bool batch, bool half, uint32_t tile, double tol) {
        dev.fill_zero(by);
        std::vector<Binding> binds{R(bw), R(half ? bxh : bx), R(bxs), W(by), R(dummy)};
        if (bt.valid()) binds.push_back(R(bt));
        Program p(dev);
        if (batch) p.dispatch(dev.pipeline(kernel), binds, &pb, sizeof pb, stride / 64, (NT + tile - 1) / tile);
        else p.dispatch(dev.pipeline(kernel), binds, &p1, sizeof p1, stride / 64);
        p.finish(); p.run();
        size_t n = batch ? got.size() : rows;
        dev.download(by, got.data(), n * 4);
        double err = 0, norm = 0;
        for (size_t i = 0; i < n; i++) { double d = (double)got[i] - want[i]; err += d * d; norm += (double)want[i] * want[i]; }
        double rel = std::sqrt(err / std::max(norm, 1e-30));
        bool ok = rel < tol;
        printf("%s %-7s %-28s %u rows x %u columns, relative error %.2e\n", ok ? "ok  " : "FAIL", ti->name, what, rows, n_in, rel);
        failed += !ok;
    };
    check("single vector", pf->kernel, false, false, 1, 1e-4);
    check("batched (16 vectors)", "matmat" + base, true, false, 4, 1e-4);
    if (n_in % 256 == 0) {
        // half-precision weights and inputs: expect about 1e-3
        check("matrix path, emulated", "wmmaemu" + base, true, true, 16, 5e-3);
        if (dev.caps().coop_matrix) check("matrix instructions", "wmma" + base, true, true, 16, 5e-3);
        else printf("     %-7s matrix instructions          skipped: the device does not have them\n", ti->name);
    }
    (void)ref_norm;
    return failed ? 1 : 0;
}

// Build every kernel on the selected device and report what the driver compiled it to.
static int cmd_kernels(int argc, char **argv) {
    std::string only, device; bool code = false; uint32_t subgroup = 0; std::vector<uint32_t> spec;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--code") code = true;
        else if (a == "--wave") subgroup = (uint32_t)atoi(next());
        else if (a == "--device") device = next();
        else if (a == "--spec") { for (char *t = strtok((char *)next(), ","); t; t = strtok(nullptr, ",")) spec.push_back((uint32_t)atoi(t)); }
        else only = a;
    }
    Device dev(device);
    printf("device %s (driver %s)\n", dev.caps().name.c_str(), dev.caps().driver.c_str());
    int failed = 0;
    for (const ShaderBlob &b : all_shaders()) {
        if (!only.empty() && only != b.name) continue;
        if (!strncmp(b.name, "wmma_", 5) && !dev.caps().coop_matrix) { printf("== %s: skipped (the device has no cooperative-matrix support)\n", b.name); continue; }
        // Kernels whose workgroup size is a specialization constant need one to compile.
        std::vector<uint32_t> sp = spec;
        if (sp.empty() && (!strcmp(b.name, "attn_prep") || !strncmp(b.name, "dn_", 3))) sp = {128};
        try {
            std::string info = dev.describe_pipeline(b.name, sp, subgroup, code);
            dev.pipeline(b.name, sp, subgroup);
            printf("== %s: ok\n%s", b.name, info.c_str());
        } catch (const std::exception &e) { printf("== %s: FAILED: %s\n", b.name, e.what()); failed++; }
    }
    return failed ? 1 : 0;
}

int main(int argc, char **argv) {
    try {
        if (argc >= 2 && !strcmp(argv[1], "info")) return cmd_info(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "plan")) return cmd_plan(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "dequant")) return cmd_dequant(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "run")) return cmd_run(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "chat")) return cmd_chat(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "compare")) return cmd_compare(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "devices")) return cmd_devices();
        if (argc >= 2 && !strcmp(argv[1], "kernels")) return cmd_kernels(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "bench")) return cmd_bench(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "test-matvec")) return cmd_test_matvec(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "tokenize")) return cmd_tokenize(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "dequant-raw")) return cmd_dequant_raw(argc - 2, argv + 2);
        fprintf(stderr, "usage: gfxi <command> ...\n  info <model.gguf> [--tensors]   show model metadata and weight encodings\n"
                        "  plan <model.gguf> [--vram GiB]  GPU memory needed and how much context fits\n"
                        "  run <model.gguf> -p <prompt>    generate text\n"
                        "  chat <model.gguf>               interactive chat, one line per turn\n"
                        "  devices                         list Vulkan devices and the features this engine uses\n"
                        "  compare <model.gguf>            check the GPU engine against the CPU reference\n"
                        "  bench [model.gguf] [--wave 32|64]  memory roofline and kernel throughput; with a model, prompt and generation tok/s\n"
                        "  kernels [name] [--code]         compile every kernel on the device; --code prints the machine code\n"
                        "  tokenize <model.gguf> <text|@file>  print token ids\n");
        return 2;
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
