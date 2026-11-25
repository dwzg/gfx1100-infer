#include "vk.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace gfx {

namespace {
constexpr uint32_t k_max_bindings = 12;
constexpr uint32_t k_push_bytes = 128;
constexpr uint64_t k_staging_bytes = 64ull << 20;

[[noreturn]] void vk_fail(const char *what, VkResult r) {
    throw std::runtime_error(std::string("Vulkan: ") + what + " failed (code " + std::to_string((int)r) + ")");
}
void check(VkResult r, const char *what) { if (r != VK_SUCCESS) vk_fail(what, r); }
std::string lower(std::string s) { for (auto &c : s) c = (char)tolower((unsigned char)c); return s; }

int device_rank(const VkPhysicalDeviceProperties &p) {
    int r = 0;
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) r += 100;
    else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) r += 50;
    else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) r -= 50;
    if (p.vendorID == 0x1002) r += 20;       // AMD
    if (p.deviceID == 0x744C) r += 10;       // Navi 31 (RX 7900 XTX / XT)
    return r;
}
}  // namespace

std::vector<std::string> Device::list_devices() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return {};
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<std::string> out;
    for (auto pd : pds) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd, &p); out.emplace_back(p.deviceName); }
    vkDestroyInstance(inst, nullptr);
    return out;
}

Device::Device(const std::string &select_in) {
    std::string select = select_in;
    if (const char *e = getenv("GFX_DEVICE"); e && *e) select = e;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gfx1100-infer"; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    check(vkCreateInstance(&ici, nullptr, &inst_), "vkCreateInstance");

    uint32_t n = 0; vkEnumeratePhysicalDevices(inst_, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst_, &n, pds.data());
    if (pds.empty()) throw std::runtime_error("no Vulkan device found (is a Vulkan driver installed?)");
    int best = -1, best_rank = -1000;
    for (int i = 0; i < (int)pds.size(); i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        if (VK_API_VERSION_MINOR(p.apiVersion) < 3 && VK_API_VERSION_MAJOR(p.apiVersion) == 1) continue;
        if (!select.empty()) {
            bool is_index = select.find_first_not_of("0123456789") == std::string::npos;
            if (is_index ? atoi(select.c_str()) != i : lower(p.deviceName).find(lower(select)) == std::string::npos) continue;
        }
        int r = device_rank(p);
        if (r > best_rank) { best_rank = r; best = i; }
    }
    if (best < 0) throw std::runtime_error(select.empty() ? "no Vulkan 1.3 device found" : "no Vulkan 1.3 device matches '" + select + "'");
    phys_ = pds[best];

    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys_, &props);
    uint32_t n_ext = 0; vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n_ext, nullptr);
    std::vector<VkExtensionProperties> exts(n_ext); vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n_ext, exts.data());
    auto has_ext = [&](const char *name) { for (auto &e : exts) if (!strcmp(e.extensionName, name)) return true; return false; };

    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &p11; p11.pNext = &p12; p12.pNext = &p13;
    vkGetPhysicalDeviceProperties2(phys_, &p2);

    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fpe{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    f2.pNext = &f11; f11.pNext = &f12; f12.pNext = &f13;
    void **tail = &f13.pNext;
    bool ext_coop = has_ext(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME), ext_pe = has_ext(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    if (ext_coop) { *tail = &fcm; tail = &fcm.pNext; }
    if (ext_pe) { *tail = &fpe; tail = &fpe.pNext; }
    vkGetPhysicalDeviceFeatures2(phys_, &f2);

    caps_.name = props.deviceName; caps_.driver = p12.driverName; caps_.vendor_id = props.vendorID; caps_.device_id = props.deviceID;
    caps_.is_radv = p12.driverID == VK_DRIVER_ID_MESA_RADV; caps_.is_cpu = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    caps_.subgroup_size = p11.subgroupSize; caps_.min_subgroup = p13.minSubgroupSize; caps_.max_subgroup = p13.maxSubgroupSize;
    caps_.subgroup_size_control = f13.subgroupSizeControl && (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
    caps_.max_alloc = p11.maxMemoryAllocationSize; caps_.max_buffer = p13.maxBufferSize;
    caps_.max_storage_range = props.limits.maxStorageBufferRange; caps_.storage_align = props.limits.minStorageBufferOffsetAlignment;
    caps_.fp16 = f12.shaderFloat16; caps_.int8 = f12.shaderInt8; caps_.int16 = f2.features.shaderInt16;
    caps_.storage8 = f12.storageBuffer8BitAccess; caps_.storage16 = f11.storageBuffer16BitAccess;
    caps_.int_dot = f13.shaderIntegerDotProduct; caps_.int_dot_4x8_mixed_accel = p13.integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    caps_.coop_matrix = ext_coop && fcm.cooperativeMatrix; caps_.exec_props = ext_pe && fpe.pipelineExecutableInfo;
    caps_.memory_budget = has_ext(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) caps_.vram_bytes = std::max<uint64_t>(caps_.vram_bytes, mp.memoryHeaps[i].size);

    if (!has_ext(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) throw std::runtime_error(caps_.name + ": VK_KHR_push_descriptor is required");
    if (!caps_.storage16 || !caps_.fp16 || !caps_.int16 || !caps_.int8)
        throw std::runtime_error(caps_.name + ": 16-bit storage, 16-bit float and 8/16-bit integer shader support are required");

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nq, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < nq && !found; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queue_family_ = i; found = true; }
    if (!found) throw std::runtime_error("device has no compute queue");

    // Enable exactly the features the kernels use.
    VkPhysicalDeviceFeatures2 e2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features e13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR ecm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR epe{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    e2.features.shaderInt16 = VK_TRUE; e2.features.shaderInt64 = f2.features.shaderInt64;
    e11.storageBuffer16BitAccess = VK_TRUE;
    e12.shaderFloat16 = VK_TRUE; e12.shaderInt8 = VK_TRUE; e12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
    e13.shaderIntegerDotProduct = f13.shaderIntegerDotProduct; e13.subgroupSizeControl = f13.subgroupSizeControl;
    e13.computeFullSubgroups = f13.computeFullSubgroups; e13.maintenance4 = f13.maintenance4;
    e2.pNext = &e11; e11.pNext = &e12; e12.pNext = &e13; tail = &e13.pNext;
    std::vector<const char *> enabled{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    if (caps_.coop_matrix) { ecm.cooperativeMatrix = VK_TRUE; *tail = &ecm; tail = &ecm.pNext; enabled.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME); }
    if (caps_.exec_props) { epe.pipelineExecutableInfo = VK_TRUE; *tail = &epe; tail = &epe.pNext; enabled.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME); }
    if (caps_.memory_budget) enabled.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; dq.queueFamilyIndex = queue_family_; dq.queueCount = 1; dq.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &e2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dq;
    dci.enabledExtensionCount = (uint32_t)enabled.size(); dci.ppEnabledExtensionNames = enabled.data();
    check(vkCreateDevice(phys_, &dci, nullptr, &dev_), "vkCreateDevice");
    vkGetDeviceQueue(dev_, queue_family_, 0, &queue_);
    push_descriptor_ = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(dev_, "vkCmdPushDescriptorSetKHR");
    if (!push_descriptor_) throw std::runtime_error("vkCmdPushDescriptorSetKHR is not available");

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = queue_family_;
    check(vkCreateCommandPool(dev_, &cpi, nullptr, &pool_), "vkCreateCommandPool");

    std::vector<VkDescriptorSetLayoutBinding> b(k_max_bindings);
    for (uint32_t i = 0; i < k_max_bindings; i++) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR; dli.bindingCount = k_max_bindings; dli.pBindings = b.data();
    check(vkCreateDescriptorSetLayout(dev_, &dli, nullptr, &set_layout_), "vkCreateDescriptorSetLayout");
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, k_push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout_; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    check(vkCreatePipelineLayout(dev_, &pli, nullptr, &layout_), "vkCreatePipelineLayout");
    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    check(vkCreatePipelineCache(dev_, &pci, nullptr, &cache_), "vkCreatePipelineCache");

}

Device::~Device() {
    // RADV's hardware-less "null" device (RADV_FORCE_FAMILY, used to inspect generated code) cannot
    // execute or tear anything down; leave it to process exit.
    if (caps_.name.rfind("Null hardware", 0) == 0) return;
    if (dev_) {
        vkDeviceWaitIdle(dev_);
        for (auto &[k, p] : pipelines_) vkDestroyPipeline(dev_, p->handle, nullptr);
        for (auto &[k, m] : modules_) vkDestroyShaderModule(dev_, m, nullptr);
        for (auto *arenas : {&device_arenas_, &host_arenas_})
            for (auto &a : *arenas) { vkDestroyBuffer(dev_, a.buf, nullptr); vkFreeMemory(dev_, a.mem, nullptr); }
        if (cache_) vkDestroyPipelineCache(dev_, cache_, nullptr);
        if (layout_) vkDestroyPipelineLayout(dev_, layout_, nullptr);
        if (set_layout_) vkDestroyDescriptorSetLayout(dev_, set_layout_, nullptr);
        if (pool_) vkDestroyCommandPool(dev_, pool_, nullptr);
        vkDestroyDevice(dev_, nullptr);
    }
    if (inst_) vkDestroyInstance(inst_, nullptr);
}

uint32_t Device::find_memory_type(uint32_t type_bits, bool host) const {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    // Host buffers are read back by the CPU every token: prefer cached system memory over a
    // write-combined mapping of VRAM. Device buffers: plain device-local memory.
    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    std::vector<std::pair<VkMemoryPropertyFlags, VkMemoryPropertyFlags>> prefs;  // (required, forbidden)
    if (host) prefs = {{hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}, {hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0}, {hv, 0}};
    else prefs = {{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT}, {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0}, {0, 0}};
    for (auto [req, forbid] : prefs)
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & req) == req && !(mp.memoryTypes[i].propertyFlags & forbid)) return i;
    throw std::runtime_error("no suitable Vulkan memory type");
}

Device::Arena Device::new_arena(uint64_t size, bool host) {
    Arena a{}; a.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(dev_, &bci, nullptr, &a.buf), "vkCreateBuffer");
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev_, a.buf, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = req.size; mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, host);
    VkResult r = vkAllocateMemory(dev_, &mai, nullptr, &a.mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(dev_, a.buf, nullptr);
        throw std::runtime_error("out of " + std::string(host ? "host" : "GPU") + " memory allocating " + std::to_string(size >> 20) + " MiB (" +
                                 std::to_string(allocated_ >> 20) + " MiB already in use)");
    }
    check(vkBindBufferMemory(dev_, a.buf, a.mem, 0), "vkBindBufferMemory");
    if (host) check(vkMapMemory(dev_, a.mem, 0, VK_WHOLE_SIZE, 0, &a.mapped), "vkMapMemory");
    allocated_ += size;
    return a;
}

Buffer Device::alloc_from(std::vector<Arena> &arenas, uint64_t size, bool host) {
    if (size == 0) size = 4;
    uint64_t align = std::max<uint64_t>(caps_.storage_align, 256);
    size = (size + 255) & ~255ull;
    if (size > caps_.max_storage_range) throw std::runtime_error("internal: buffer of " + std::to_string(size) + " bytes exceeds the device's storage binding limit");
    for (auto &a : arenas) {
        uint64_t off = (a.used + align - 1) / align * align;
        if (off + size <= a.size) { a.used = off + size; return Buffer{a.buf, off, size, a.mapped ? (char *)a.mapped + off : nullptr}; }
    }
    uint64_t block = host ? (64ull << 20) : (512ull << 20);
    block = std::min<uint64_t>(block, std::min(caps_.max_alloc, caps_.max_buffer) / 2);
    arenas.push_back(new_arena(std::max(size, block), host));
    Arena &a = arenas.back(); a.used = size;
    return Buffer{a.buf, 0, size, a.mapped};
}
Buffer Device::alloc(uint64_t size) { return alloc_from(device_arenas_, size, false); }
Buffer Device::alloc_host(uint64_t size) { return alloc_from(host_arenas_, size, true); }

void Device::upload(const Buffer &dst, const void *data, uint64_t size) {
    if (size > dst.size) throw std::runtime_error("internal: upload larger than buffer");
    if (dst.mapped) { memcpy(dst.mapped, data, size); return; }
    if (!staging_.valid()) staging_ = alloc_host(k_staging_bytes);
    for (uint64_t done = 0; done < size;) {
        uint64_t n = std::min(size - done, staging_.size);
        memcpy(staging_.mapped, (const char *)data + done, n);
        Program p(*this); p.copy(staging_.sub(0, n), dst.sub(done, n)); p.finish(); p.run();
        done += n;
    }
}
void Device::download(const Buffer &src, void *data, uint64_t size) {
    if (size > src.size) throw std::runtime_error("internal: download larger than buffer");
    if (src.mapped) { memcpy(data, src.mapped, size); return; }
    if (!staging_.valid()) staging_ = alloc_host(k_staging_bytes);
    for (uint64_t done = 0; done < size;) {
        uint64_t n = std::min(size - done, staging_.size);
        Program p(*this); p.copy(src.sub(done, n), staging_.sub(0, n)); p.finish(); p.run();
        memcpy((char *)data + done, staging_.mapped, n);
        done += n;
    }
}
void Device::fill_zero(const Buffer &dst) {
    if (dst.mapped) { memset(dst.mapped, 0, dst.size); return; }
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool = pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(dev_, &ai, &cmd), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdFillBuffer(cmd, dst.handle, dst.offset, dst.size, 0);   // sizes are multiples of 256, as required
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    check(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    check(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    vkFreeCommandBuffers(dev_, pool_, 1, &cmd);
}

VkPipeline Device::build_pipeline(const std::string &shader, const std::vector<uint32_t> &spec, uint32_t subgroup, VkPipelineCreateFlags flags) {
    VkShaderModule &mod = modules_[shader];
    if (!mod) {
        const ShaderBlob &blob = find_shader(shader);
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = blob.bytes; smi.pCode = blob.code;
        check(vkCreateShaderModule(dev_, &smi, nullptr, &mod), "vkCreateShaderModule");
    }
    std::vector<VkSpecializationMapEntry> entries(spec.size());
    for (uint32_t i = 0; i < spec.size(); i++) entries[i] = {i, i * 4, 4};
    VkSpecializationInfo si{(uint32_t)entries.size(), entries.data(), spec.size() * 4, spec.data()};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.flags = flags; cpi.layout = layout_;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod; cpi.stage.pName = "main"; cpi.stage.pSpecializationInfo = spec.empty() ? nullptr : &si;
    if (subgroup && caps_.subgroup_size_control && subgroup >= caps_.min_subgroup && subgroup <= caps_.max_subgroup) {
        rss.requiredSubgroupSize = subgroup; cpi.stage.pNext = &rss;
    }
    VkPipeline pipe;
    VkResult r = vkCreateComputePipelines(dev_, cache_, 1, &cpi, nullptr, &pipe);
    if (r != VK_SUCCESS) throw std::runtime_error("the driver rejected kernel '" + shader + "' (code " + std::to_string((int)r) + ")");
    return pipe;
}

const Pipeline *Device::pipeline(const std::string &shader, const std::vector<uint32_t> &spec, uint32_t subgroup) {
    std::string key = shader + "/" + std::to_string(subgroup);
    for (uint32_t s : spec) key += ":" + std::to_string(s);
    auto &slot = pipelines_[key];
    if (!slot) {
        slot = std::make_unique<Pipeline>();
        slot->name = shader;
        slot->handle = build_pipeline(shader, spec, subgroup, 0);
    }
    return slot.get();
}

std::string Device::describe_pipeline(const std::string &shader, const std::vector<uint32_t> &spec, uint32_t subgroup, bool with_code) {
    if (!caps_.exec_props) return "(this driver does not report pipeline statistics)\n";
    auto get_exe = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(dev_, "vkGetPipelineExecutablePropertiesKHR");
    auto get_stats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(dev_, "vkGetPipelineExecutableStatisticsKHR");
    auto get_ir = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(dev_, "vkGetPipelineExecutableInternalRepresentationsKHR");
    VkPipeline pipe = build_pipeline(shader, spec, subgroup, VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR);
    std::string out;
    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR}; pi.pipeline = pipe;
    uint32_t nx = 0; get_exe(dev_, &pi, &nx, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> xp(nx, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    get_exe(dev_, &pi, &nx, xp.data());
    for (uint32_t x = 0; x < nx; x++) {
        out += "wave size " + std::to_string(xp[x].subgroupSize) + "\n";
        VkPipelineExecutableInfoKHR xi{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR}; xi.pipeline = pipe; xi.executableIndex = x;
        uint32_t ns = 0; get_stats(dev_, &xi, &ns, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        get_stats(dev_, &xi, &ns, st.data());
        for (auto &s : st) {
            if (!strcmp(s.name, "Driver pipeline hash") || !strcmp(s.name, "Hash")) continue;
            out += std::string("  ") + s.name + " = " + std::to_string(s.value.u64) + "\n";
        }
        if (!with_code) continue;
        uint32_t nr = 0; get_ir(dev_, &xi, &nr, nullptr);
        std::vector<VkPipelineExecutableInternalRepresentationKHR> ir(nr, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
        get_ir(dev_, &xi, &nr, ir.data());
        std::vector<std::vector<char>> data(nr);
        for (uint32_t i = 0; i < nr; i++) { data[i].resize(ir[i].dataSize); ir[i].pData = data[i].data(); }
        get_ir(dev_, &xi, &nr, ir.data());
        for (uint32_t i = 0; i < nr; i++)
            if (strstr(ir[i].name, "ssembly")) out += std::string(data[i].data(), strnlen(data[i].data(), data[i].size())) + "\n";
    }
    vkDestroyPipeline(dev_, pipe, nullptr);
    return out;
}

// ---- Program ----

Program::Program(Device &dev) : dev_(dev) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = dev_.pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(dev_.dev_, &ai, &cmd_), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(dev_.dev_, &fi, nullptr, &fence_), "vkCreateFence");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(cmd_, &bi), "vkBeginCommandBuffer");
}
Program::~Program() {
    vkDestroyFence(dev_.dev_, fence_, nullptr);
    vkFreeCommandBuffers(dev_.dev_, dev_.pool_, 1, &cmd_);
}

bool Program::overlaps(const std::vector<Range> &set, const Range &r) {
    for (const Range &s : set) if (s.b == r.b && s.lo < r.hi && r.lo < s.hi) return true;
    return false;
}

void Program::barrier() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    mb.dstAccessMask = mb.srcAccessMask;
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cmd_, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr);
    pending_reads_.clear(); pending_writes_.clear(); n_barrier_++;
}

void Program::dispatch(const Pipeline *p, const std::vector<Binding> &bindings, const void *pc, size_t pc_bytes, uint32_t gx, uint32_t gy, uint32_t gz) {
    if (finished_) throw std::runtime_error("internal: program already finished");
    if (bindings.size() > k_max_bindings || pc_bytes > k_push_bytes) throw std::runtime_error("internal: too many bindings or push constants");
    if (gx == 0 || gy == 0 || gz == 0) return;
    // A dispatch must wait for earlier un-synchronised work if it reads what that work writes,
    // or writes what that work reads or writes.
    bool conflict = false;
    for (const Binding &b : bindings) {
        Range r{b.buf.handle, b.buf.offset, b.buf.offset + b.buf.size};
        if (overlaps(pending_writes_, r) || (b.write && overlaps(pending_reads_, r))) { conflict = true; break; }
    }
    if (conflict) barrier();
    for (const Binding &b : bindings) (b.write ? pending_writes_ : pending_reads_).push_back({b.buf.handle, b.buf.offset, b.buf.offset + b.buf.size});

    if (bound_ != p) { vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p->handle); bound_ = p; }
    VkDescriptorBufferInfo infos[k_max_bindings]; VkWriteDescriptorSet writes[k_max_bindings];
    for (size_t i = 0; i < bindings.size(); i++) {
        infos[i] = {bindings[i].buf.handle, bindings[i].buf.offset, bindings[i].buf.size};
        writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstBinding = (uint32_t)i; writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i];
    }
    dev_.push_descriptor_(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, dev_.layout_, 0, (uint32_t)bindings.size(), writes);
    if (pc_bytes) vkCmdPushConstants(cmd_, dev_.layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pc_bytes, pc);
    vkCmdDispatch(cmd_, gx, gy, gz);
    n_dispatch_++;
}

void Program::copy(const Buffer &src, const Buffer &dst) {
    Range rs{src.handle, src.offset, src.offset + src.size}, rd{dst.handle, dst.offset, dst.offset + dst.size};
    if (overlaps(pending_writes_, rs) || overlaps(pending_writes_, rd) || overlaps(pending_reads_, rd)) barrier();
    pending_reads_.push_back(rs); pending_writes_.push_back(rd);
    VkBufferCopy region{src.offset, dst.offset, std::min(src.size, dst.size)};
    vkCmdCopyBuffer(cmd_, src.handle, dst.handle, 1, &region);
}

void Program::finish() {
    if (finished_) return;
    // Make results visible to the host and to later submissions.
    barrier();
    check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");
    finished_ = true;
}

void Program::run() {
    if (!finished_) finish();
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd_;
    check(vkQueueSubmit(dev_.queue_, 1, &si, fence_), "vkQueueSubmit");
    check(vkWaitForFences(dev_.dev_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    check(vkResetFences(dev_.dev_, 1, &fence_), "vkResetFences");
}

}  // namespace gfx
