#include "gguf.h"

#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gfx {

static const struct { uint32_t id; TypeInfo ti; } k_types[] = {
    {T_F32, {"F32", 1, 4, true}},       {T_F16, {"F16", 1, 2, true}},       {T_BF16, {"BF16", 1, 2, true}},
    {T_Q4_0, {"Q4_0", 32, 18, true}},   {T_Q4_1, {"Q4_1", 32, 20, true}},   {T_Q5_0, {"Q5_0", 32, 22, true}},
    {T_Q5_1, {"Q5_1", 32, 24, true}},   {T_Q8_0, {"Q8_0", 32, 34, true}},   {T_Q2_K, {"Q2_K", 256, 84, true}},
    {T_Q3_K, {"Q3_K", 256, 110, true}}, {T_Q4_K, {"Q4_K", 256, 144, true}}, {T_Q5_K, {"Q5_K", 256, 176, true}},
    {T_Q6_K, {"Q6_K", 256, 210, true}}, {T_IQ4_NL, {"IQ4_NL", 32, 18, true}}, {T_IQ4_XS, {"IQ4_XS", 256, 136, true}},
    {T_IQ3_S, {"IQ3_S", 256, 110, true}}, {T_MXFP4, {"MXFP4", 32, 17, true}}, {T_NVFP4, {"NVFP4", 64, 36, true}},
    {T_I8, {"I8", 1, 1, false}},        {T_I16, {"I16", 1, 2, false}},      {T_I32, {"I32", 1, 4, false}},
    {T_I64, {"I64", 1, 8, false}},      {T_F64, {"F64", 1, 8, false}},
    // Known ids without a decoder here (reported by name instead of as a bare number).
    {9, {"Q8_1", 32, 36, false}},       {16, {"IQ2_XXS", 256, 66, false}},  {17, {"IQ2_XS", 256, 74, false}},
    {18, {"IQ3_XXS", 256, 98, false}},  {19, {"IQ1_S", 256, 50, false}},    {22, {"IQ2_S", 256, 82, false}},
    {29, {"IQ1_M", 256, 56, false}},    {34, {"TQ1_0", 256, 54, false}},    {35, {"TQ2_0", 256, 66, false}},
    {41, {"Q1_0", 128, 18, false}},     {42, {"Q2_0", 64, 18, false}},      {43, {"F8_E4M3", 1, 1, false}},
};

const TypeInfo *type_info(uint32_t type) {
    for (const auto &t : k_types)
        if (t.id == type) return &t.ti;
    return nullptr;
}

static const uint32_t k_meta_size[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

static double load_num(const uint8_t *p, MetaType t, int64_t *iout) {
    double f = 0; int64_t i = 0;
    switch (t) {
        case M_U8: i = *p; f = (double)i; break;
        case M_I8: i = (int8_t)*p; f = (double)i; break;
        case M_U16: { uint16_t v; memcpy(&v, p, 2); i = v; f = (double)i; break; }
        case M_I16: { int16_t v; memcpy(&v, p, 2); i = v; f = (double)i; break; }
        case M_U32: { uint32_t v; memcpy(&v, p, 4); i = v; f = (double)i; break; }
        case M_I32: { int32_t v; memcpy(&v, p, 4); i = v; f = (double)i; break; }
        case M_F32: { float v; memcpy(&v, p, 4); f = v; i = (int64_t)v; break; }
        case M_BOOL: i = *p != 0; f = (double)i; break;
        case M_U64: { uint64_t v; memcpy(&v, p, 8); i = (int64_t)v; f = (double)v; break; }
        case M_I64: { int64_t v; memcpy(&v, p, 8); i = v; f = (double)v; break; }
        case M_F64: { double v; memcpy(&v, p, 8); f = v; i = (int64_t)v; break; }
        default: break;
    }
    if (iout) *iout = i;
    return f;
}

int64_t MetaValue::as_int() const { return (type == M_F32 || type == M_F64) ? (int64_t)f : i; }
double MetaValue::as_float() const { return (type == M_F32 || type == M_F64) ? f : (double)i; }
int64_t MetaValue::arr_int(uint64_t idx) const {
    if (type != M_ARR || idx >= count || !raw) throw std::runtime_error("gguf: bad array access");
    int64_t v; load_num(raw + idx * k_meta_size[elem_type], elem_type, &v); return v;
}
double MetaValue::arr_float(uint64_t idx) const {
    if (type != M_ARR || idx >= count || !raw) throw std::runtime_error("gguf: bad array access");
    return load_num(raw + idx * k_meta_size[elem_type], elem_type, nullptr);
}

namespace {
struct Cursor {
    const uint8_t *base; size_t size; size_t pos = 0;
    void need(size_t n) const {
        if (n > size || pos > size - n) throw std::runtime_error("gguf: file truncated in header");
    }
    template <class T> T rd() { need(sizeof(T)); T v; memcpy(&v, base + pos, sizeof(T)); pos += sizeof(T); return v; }
    std::string_view str() {
        uint64_t n = rd<uint64_t>(); need(n);
        std::string_view s((const char *)base + pos, n); pos += n; return s;
    }
};
}  // namespace

GGUF::~GGUF() { if (map_) munmap((void *)map_, map_size_); }

void GGUF::open(const std::string &path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path + ": " + strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("cannot stat " + path); }
    file_size = (uint64_t)st.st_size;
    void *m = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
    map_ = (const uint8_t *)m; map_size_ = file_size;

    Cursor c{map_, map_size_};
    if (c.rd<uint32_t>() != 0x46554747u) throw std::runtime_error(path + " is not a GGUF file");
    version = c.rd<uint32_t>();
    if (version < 2 || version > 3) throw std::runtime_error("unsupported GGUF version " + std::to_string(version));
    uint64_t n_tensors = c.rd<uint64_t>(), n_kv = c.rd<uint64_t>();
    if (n_tensors > (1u << 24) || n_kv > (1u << 24)) throw std::runtime_error("gguf: implausible header counts");

    meta.reserve(n_kv);
    for (uint64_t k = 0; k < n_kv; k++) {
        std::string key(c.str());
        MetaValue v;
        uint32_t t = c.rd<uint32_t>();
        if (t > M_F64) throw std::runtime_error("gguf: bad value type for key " + key);
        v.type = (MetaType)t;
        if (v.type == M_STR) {
            v.s = c.str();
        } else if (v.type == M_ARR) {
            uint32_t et = c.rd<uint32_t>();
            if (et > M_F64 || et == M_ARR) throw std::runtime_error("gguf: unsupported array type for key " + key);
            v.elem_type = (MetaType)et; v.count = c.rd<uint64_t>();
            if (v.elem_type == M_STR) {
                if (v.count > map_size_ / 8) throw std::runtime_error("gguf: bad array length for key " + key);
                v.strs.reserve(v.count);
                for (uint64_t j = 0; j < v.count; j++) v.strs.push_back(c.str());
            } else {
                uint64_t es = k_meta_size[v.elem_type];
                if (v.count > map_size_ / es) throw std::runtime_error("gguf: bad array length for key " + key);
                c.need(v.count * es); v.raw = map_ + c.pos; c.pos += v.count * es;
            }
        } else {
            c.need(k_meta_size[v.type]);
            v.f = load_num(map_ + c.pos, v.type, &v.i); v.u = (uint64_t)v.i; c.pos += k_meta_size[v.type];
        }
        meta_index_[key] = meta.size();
        meta.emplace_back(std::move(key), std::move(v));
    }
    architecture = get_str("general.architecture");
    alignment = (uint64_t)get_int("general.alignment", 32);
    if (alignment == 0 || (alignment & (alignment - 1))) throw std::runtime_error("gguf: bad alignment");

    tensors.resize(n_tensors);
    for (auto &t : tensors) {
        t.name = std::string(c.str());
        uint32_t nd = c.rd<uint32_t>();
        if (nd > 8) throw std::runtime_error("gguf: tensor " + t.name + " has too many dimensions");
        t.nelems = 1;
        for (uint32_t d = 0; d < nd; d++) {
            t.dims.push_back(c.rd<uint64_t>());
            if (t.dims.back() != 0 && t.nelems > UINT64_MAX / t.dims.back()) throw std::runtime_error("gguf: tensor size overflow");
            t.nelems *= t.dims.back();
        }
        t.type = c.rd<uint32_t>(); t.offset = c.rd<uint64_t>();
    }
    data_offset = (c.pos + alignment - 1) / alignment * alignment;
    for (size_t k = 0; k < tensors.size(); k++) {
        auto &t = tensors[k];
        const TypeInfo *ti = type_info(t.type);
        if (!ti) throw std::runtime_error("tensor " + t.name + " has unknown type id " + std::to_string(t.type));
        if (t.nelems % ti->block_elems) throw std::runtime_error("tensor " + t.name + ": size not a multiple of the block size");
        t.nbytes = t.nelems / ti->block_elems * ti->block_bytes;
        if (data_offset > file_size || t.offset > file_size - data_offset || t.nbytes > file_size - data_offset - t.offset)
            throw std::runtime_error("tensor " + t.name + " extends past the end of the file (truncated download?)");
        t.data = map_ + data_offset + t.offset;
        tensor_index_[t.name] = k;
    }
}

const MetaValue *GGUF::find(std::string_view key) const {
    auto it = meta_index_.find(key);
    return it == meta_index_.end() ? nullptr : &meta[it->second].second;
}
int64_t GGUF::get_int(std::string_view key, int64_t def) const {
    const MetaValue *v = find(key);
    return (v && v->type != M_STR && v->type != M_ARR) ? v->as_int() : def;
}
double GGUF::get_float(std::string_view key, double def) const {
    const MetaValue *v = find(key);
    return (v && v->type != M_STR && v->type != M_ARR) ? v->as_float() : def;
}
std::string GGUF::get_str(std::string_view key, std::string_view def) const {
    const MetaValue *v = find(key);
    return std::string((v && v->type == M_STR) ? v->s : def);
}
const TensorInfo *GGUF::tensor(std::string_view name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors[it->second];
}

}  // namespace gfx
