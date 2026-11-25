// Thin Vulkan compute layer: device selection, memory arenas, pipelines, and recorded programs
// (sequences of dispatches with automatically placed barriers).
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace gfx {

struct DeviceCaps {
    std::string name, driver;
    uint32_t vendor_id = 0, device_id = 0;
    bool is_radv = false, is_cpu = false;
    uint32_t subgroup_size = 0, min_subgroup = 0, max_subgroup = 0;
    bool subgroup_size_control = false;
    uint64_t max_alloc = 0, max_buffer = 0, max_storage_range = 0, storage_align = 1;
    uint64_t vram_bytes = 0;            // largest device-local heap
    bool fp16 = false, int8 = false, int16 = false, storage8 = false, storage16 = false;
    bool int_dot = false, int_dot_4x8_mixed_accel = false;
    bool coop_matrix = false;           // VK_KHR_cooperative_matrix (WMMA on RDNA3)
    bool exec_props = false;            // VK_KHR_pipeline_executable_properties
    bool memory_budget = false;
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    uint64_t offset = 0, size = 0;      // range inside `handle`
    void *mapped = nullptr;             // host buffers only
    bool valid() const { return handle != VK_NULL_HANDLE; }
    Buffer sub(uint64_t off, uint64_t len) const { return Buffer{handle, offset + off, len, mapped ? (char *)mapped + off : nullptr}; }
};

struct ShaderBlob { const char *name; const uint32_t *code; size_t bytes; };
// Looks up a kernel compiled into the binary; throws if the name is unknown.
const ShaderBlob &find_shader(const std::string &name);
const std::vector<ShaderBlob> &all_shaders();

struct Pipeline {
    VkPipeline handle = VK_NULL_HANDLE;
    std::string name;
};

struct Binding { Buffer buf; bool write; };
inline Binding R(const Buffer &b) { return {b, false}; }
inline Binding W(const Buffer &b) { return {b, true}; }

class Device;

// A recorded command sequence that can be submitted repeatedly.
class Program {
  public:
    Program(Device &dev);
    ~Program();
    Program(const Program &) = delete;
    // Bindings are pushed in order to binding points 0, 1, ...
    void dispatch(const Pipeline *p, const std::vector<Binding> &bindings, const void *pc, size_t pc_bytes, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
    // Like dispatch, with the group counts (three uint32) read from `args` at submission time.
    void dispatch_indirect(const Pipeline *p, const std::vector<Binding> &bindings, const void *pc, size_t pc_bytes, const Buffer &args);
    void copy(const Buffer &src, const Buffer &dst);
    // Several ranges in one command; each region is {offset in src, offset in dst, bytes}.
    void copy_many(const Buffer &src, const Buffer &dst, const std::vector<VkBufferCopy> &regions);
    void finish();
    void run();              // submit and wait
    size_t n_dispatches() const { return n_dispatch_; }
    size_t n_barriers() const { return n_barrier_; }
  private:
    struct Range { VkBuffer b; uint64_t lo, hi; };
    static bool overlaps(const std::vector<Range> &set, const Range &r);
    void barrier();
    void prepare(const Pipeline *p, const std::vector<Binding> &bindings, const void *pc, size_t pc_bytes);
    Device &dev_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::vector<Range> pending_reads_, pending_writes_;
    const Pipeline *bound_ = nullptr;
    size_t n_dispatch_ = 0, n_barrier_ = 0;
    bool finished_ = false;
};

class Device {
  public:
    // `select`: empty picks the best device (AMD discrete first); otherwise an index or a
    // case-insensitive substring of the device name. The GFX_DEVICE environment variable overrides it.
    explicit Device(const std::string &select = "");
    ~Device();
    Device(const Device &) = delete;

    const DeviceCaps &caps() const { return caps_; }
    static std::vector<std::string> list_devices();

    // Device memory (VRAM). Sub-allocated from large arenas; contents are undefined until written.
    Buffer alloc(uint64_t size);
    // Host-visible, persistently mapped memory for small inputs and outputs.
    Buffer alloc_host(uint64_t size);
    void upload(const Buffer &dst, const void *data, uint64_t size);
    void download(const Buffer &src, void *data, uint64_t size);
    void fill_zero(const Buffer &dst);
    uint64_t allocated_bytes() const { return allocated_; }
    // GPU memory this process can still allocate right now, as the driver reports it (it accounts
    // for other programs such as the desktop). Falls back to heap size minus our own use.
    uint64_t vram_free() const;

    // Compute pipeline for an embedded kernel. `spec` are specialization constants with ids 0, 1, ...
    // `subgroup` (0, 32 or 64) requests a wave size where the driver allows choosing one.
    const Pipeline *pipeline(const std::string &shader, const std::vector<uint32_t> &spec = {}, uint32_t subgroup = 0);
    // Statistics and disassembly the driver reports for a kernel (RADV: RDNA machine code).
    std::string describe_pipeline(const std::string &shader, const std::vector<uint32_t> &spec, uint32_t subgroup, bool with_code);

    VkDevice vk() const { return dev_; }

  private:
    friend class Program;
    struct Arena { VkBuffer buf; VkDeviceMemory mem; uint64_t size, used; void *mapped; };
    Buffer alloc_from(std::vector<Arena> &arenas, uint64_t size, bool host);
    Arena new_arena(uint64_t size, bool host);
    uint32_t find_memory_type(uint32_t type_bits, bool host) const;
    VkPipeline build_pipeline(const std::string &shader, const std::vector<uint32_t> &spec, uint32_t subgroup, VkPipelineCreateFlags flags);

    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    PFN_vkCmdPushDescriptorSetKHR push_descriptor_ = nullptr;
    DeviceCaps caps_;
    std::vector<Arena> device_arenas_, host_arenas_;
    Buffer staging_;
    std::map<std::string, std::unique_ptr<Pipeline>> pipelines_;
    std::map<std::string, VkShaderModule> modules_;
    uint64_t allocated_ = 0;
};

}  // namespace gfx
