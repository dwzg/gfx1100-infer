// GGUF container reader (mmap-backed, zero-copy tensor access).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace gfx {

// Tensor element types, numbered as in the GGUF specification.
enum GType : uint32_t {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ4_XS = 23,
    T_I8 = 24, T_I16 = 25, T_I32 = 26, T_I64 = 27, T_F64 = 28, T_BF16 = 30,
    T_MXFP4 = 39, T_NVFP4 = 40,
};

struct TypeInfo {
    const char *name;
    uint32_t block_elems;  // elements per block
    uint32_t block_bytes;  // bytes per block
    bool decodable;        // quant.cpp can decode it to float
    double bpw() const { return 8.0 * block_bytes / block_elems; }
};
// Returns nullptr for type ids this engine does not know.
const TypeInfo *type_info(uint32_t type);

// Metadata value types, numbered as in the GGUF specification.
enum MetaType : uint32_t {
    M_U8 = 0, M_I8, M_U16, M_I16, M_U32, M_I32, M_F32, M_BOOL, M_STR, M_ARR, M_U64, M_I64, M_F64,
};

struct MetaValue {
    MetaType type = M_U32;
    // scalars
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0;
    std::string_view s;
    // arrays
    MetaType elem_type = M_U32;
    uint64_t count = 0;
    const uint8_t *raw = nullptr;          // numeric arrays: little-endian packed elements
    std::vector<std::string_view> strs;    // string arrays

    bool is_int() const { return type <= M_I32 || type == M_U64 || type == M_I64 || type == M_BOOL; }
    int64_t as_int() const;
    double as_float() const;
    // numeric array element access (converted)
    int64_t arr_int(uint64_t idx) const;
    double arr_float(uint64_t idx) const;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;  // dims[0] is the innermost (fastest varying) dimension
    uint32_t type = 0;
    uint64_t offset = 0;         // relative to the data section
    const uint8_t *data = nullptr;
    uint64_t nbytes = 0;
    uint64_t nelems = 0;
    uint64_t dim(size_t k) const { return k < dims.size() ? dims[k] : 1; }
};

class GGUF {
  public:
    GGUF() = default;
    ~GGUF();
    GGUF(const GGUF &) = delete;
    GGUF &operator=(const GGUF &) = delete;

    // Throws std::runtime_error with a readable message on malformed input.
    // allow_truncated: accept a file that ends before its tensor data (for example a downloaded
    // header); tensors that are not fully present get data == nullptr.
    void open(const std::string &path, bool allow_truncated = false);

    const MetaValue *find(std::string_view key) const;
    // Typed getters with defaults; key may be given without the "<arch>." prefix via arch_key().
    int64_t get_int(std::string_view key, int64_t def) const;
    double get_float(std::string_view key, double def) const;
    std::string get_str(std::string_view key, std::string_view def = {}) const;
    bool has(std::string_view key) const { return find(key) != nullptr; }
    std::string arch_key(std::string_view suffix) const { return architecture + "." + std::string(suffix); }

    const TensorInfo *tensor(std::string_view name) const;

    std::string architecture;
    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t data_offset = 0;
    uint64_t file_size = 0;
    std::vector<std::pair<std::string, MetaValue>> meta;  // in file order
    std::vector<TensorInfo> tensors;                      // in file order

  private:
    const uint8_t *map_ = nullptr;
    size_t map_size_ = 0;
    std::map<std::string, size_t, std::less<>> meta_index_, tensor_index_;
};

}  // namespace gfx
