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
        fprintf(stderr, "usage: gfxi run <model.gguf> -p <prompt | @file> [-n tokens] [--chat] [--temp T] [--top-k K] [--top-p P]\n"
                        "                [--seed S] [--ctx N] [--threads N] [--device gpu|cpu|<name>] [--logits out.f32]\n");
        return 2;
    }
    std::string prompt, logits_path, device = "gpu"; int n_predict = 64, n_ctx = 0, threads = (int)std::thread::hardware_concurrency(); bool chat = false;
    SamplerParams sp; sp.temperature = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "-p") prompt = read_arg_text(next());
        else if (a == "-n") n_predict = atoi(next());
        else if (a == "--chat") chat = true;
        else if (a == "--temp") sp.temperature = (float)atof(next());
        else if (a == "--top-k") sp.top_k = atoi(next());
        else if (a == "--top-p") sp.top_p = (float)atof(next());
        else if (a == "--seed") sp.seed = strtoull(next(), nullptr, 10);
        else if (a == "--ctx") n_ctx = atoi(next());
        else if (a == "--threads") threads = atoi(next());
        else if (a == "--logits") logits_path = next();
        else if (a == "--device") device = next();
        else throw std::runtime_error("unknown option " + a);
    }
    GGUF g; g.open(argv[0]);
    ModelDesc desc = describe_model(g);
    Tokenizer tok(g);
    if (chat) prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> ids = tok.encode(prompt, true, true);
    if (ids.empty()) throw std::runtime_error("the prompt is empty");
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
        gpu = std::make_unique<GpuModel>(*dev, g, desc, n_ctx);
        fprintf(stderr, "[%s] weights %.2f GiB, context state %.2f GiB, %zu dispatches and %zu sync points per token, %d prompt tokens\n", dev->caps().name.c_str(),
                gpu->weight_bytes() / 1073741824.0, gpu->state_bytes() / 1073741824.0, gpu->dispatches_per_token(), gpu->barriers_per_token(), (int)ids.size());
        eval = [&](const int32_t *t, int n) -> const std::vector<float> & { return gpu->eval(t, n); };
    }
    Sampler sampler(sp);
    auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> *logits = &eval(ids.data(), (int)ids.size());
    auto t1 = std::chrono::steady_clock::now();
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

// Run the GPU engine and the CPU reference on the same tokens and report how closely they agree.
static int cmd_compare(int argc, char **argv) {
    if (argc < 1) { fprintf(stderr, "usage: gfxi compare <model.gguf> [-p prompt] [-n extra tokens] [--device name] [--tol T]\n"); return 2; }
    std::string prompt = "The quick brown fox jumps over the lazy dog. In 1969, humans first", device; int n_extra = 8; double tol = 0.05; bool fp16 = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "-p") prompt = read_arg_text(next());
        else if (a == "-n") n_extra = atoi(next());
        else if (a == "--device") device = next();
        else if (a == "--tol") tol = atof(next());
        else if (a == "--fp16") fp16 = true;
        else throw std::runtime_error("unknown option " + a);
    }
    GGUF g; g.open(argv[0]);
    ModelDesc desc = describe_model(g);
    Tokenizer tok(g);
    std::vector<int32_t> ids = tok.encode(prompt, true, true);
    int n_ctx = (int)ids.size() + n_extra + 1;
    Device dev(device);
    GpuModel gpu(dev, g, desc, n_ctx, fp16);
    printf("weights on device: %.3f GiB\n", gpu.weight_bytes() / 1073741824.0);
    RefModel ref(g, desc, n_ctx, (int)std::thread::hardware_concurrency());
    printf("device %s, model %s, %zu prompt tokens + %d generated\n", dev.caps().name.c_str(), desc.name.c_str(), ids.size(), n_extra);
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
    printf("  GPU memory            %.1f GiB\n", c.vram_bytes / 1073741824.0);
    printf("  wave size             %u (selectable %u..%u: %s)\n", c.subgroup_size, c.min_subgroup, c.max_subgroup, c.subgroup_size_control ? "yes" : "no");
    printf("  matrix instructions   %s\n", c.coop_matrix ? "yes (VK_KHR_cooperative_matrix)" : "no");
    printf("  packed integer dot    %s\n", c.int_dot_4x8_mixed_accel ? "yes, hardware accelerated" : c.int_dot ? "supported, not accelerated" : "no");
    printf("  largest buffer bind   %.2f GiB\n", c.max_storage_range / 1073741824.0);
    return 0;
}

// Check one quantised matvec kernel against the scalar decoder, on raw block data of the given
// type laid out as a matrix with n_in columns and a random input vector.
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
    uint64_t row_bytes = (uint64_t)n_in / ti->block_elems * ti->block_bytes;
    uint32_t rows = (uint32_t)(in.size() / row_bytes), stride = (rows + 63) / 64 * 64;
    if (rows == 0) { fprintf(stderr, "not enough data for one row\n"); return 1; }
    std::vector<float> x(n_in), xs(n_in / 16), want(rows), wrow(n_in), got(rows);
    uint32_t seed = 12345;
    for (auto &v : x) { seed = seed * 1664525u + 1013904223u; v = ((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
    for (uint32_t b = 0; b < n_in / 16; b++) { float t = 0; for (int i = 0; i < 16; i++) t += x[16 * b + i]; xs[b] = t; }
    double ref_norm = 0;
    for (uint32_t row = 0; row < rows; row++) {
        dequantize(type, in.data() + row * row_bytes, wrow.data(), n_in);
        double t = 0; for (uint32_t i = 0; i < n_in; i++) t += (double)wrow[i] * x[i];
        want[row] = (float)t; ref_norm += t * t;
    }
    Device dev(device);
    uint64_t words = (uint64_t)(n_in / pf->sb_cols) * pf->sb_words * stride;
    std::vector<uint32_t> packed(words);
    pack_rows(*pf, in.data(), n_in, 0, rows, stride, packed.data());
    Buffer bw = dev.alloc(words * 4), bx = dev.alloc(n_in * 4ull), bxs = dev.alloc(xs.size() * 4), by = dev.alloc(stride * 4ull), dummy = dev.alloc(256);
    dev.upload(bw, packed.data(), words * 4); dev.upload(bx, x.data(), n_in * 4ull); dev.upload(bxs, xs.data(), xs.size() * 4); dev.fill_zero(dummy);
    struct { uint32_t n_sb, stride, n_rows, row0, flags; } pc{n_in / pf->sb_cols, stride, rows, 0, 0};
    Program p(dev);
    p.dispatch(dev.pipeline(pf->kernel), {R(bw), R(bx), R(bxs), W(by), R(dummy)}, &pc, sizeof pc, stride / 64);
    p.finish(); p.run();
    dev.download(by, got.data(), rows * 4ull);
    double err = 0; for (uint32_t i = 0; i < rows; i++) { double d = (double)got[i] - want[i]; err += d * d; }
    double rel = std::sqrt(err / std::max(ref_norm, 1e-30));
    printf("%s %s: %u rows x %u columns, relative error %.2e\n", rel < 1e-4 ? "ok  " : "FAIL", ti->name, rows, n_in, rel);
    return rel < 1e-4 ? 0 : 1;
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
        if (argc >= 2 && !strcmp(argv[1], "dequant")) return cmd_dequant(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "run")) return cmd_run(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "compare")) return cmd_compare(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "devices")) return cmd_devices();
        if (argc >= 2 && !strcmp(argv[1], "kernels")) return cmd_kernels(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "test-matvec")) return cmd_test_matvec(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "tokenize")) return cmd_tokenize(argc - 2, argv + 2);
        if (argc >= 2 && !strcmp(argv[1], "dequant-raw")) return cmd_dequant_raw(argc - 2, argv + 2);
        fprintf(stderr, "usage: gfxi <command> ...\n  info <model.gguf> [--tensors]   show model metadata and weight encodings\n"
                        "  run <model.gguf> -p <prompt>    generate text\n"
                        "  devices                         list Vulkan devices and the features this engine uses\n"
                        "  compare <model.gguf>            check the GPU engine against the CPU reference\n"
                        "  kernels [name] [--code]         compile every kernel on the device; --code prints the machine code\n"
                        "  tokenize <model.gguf> <text|@file>  print token ids\n");
        return 2;
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
