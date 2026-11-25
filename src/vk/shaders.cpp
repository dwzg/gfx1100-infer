#include <stdexcept>

#include "vk.h"

namespace gfx {

const ShaderBlob &find_shader(const std::string &name) {
    for (const ShaderBlob &b : all_shaders())
        if (name == b.name) return b;
    throw std::runtime_error("internal: unknown kernel '" + name + "'");
}

}  // namespace gfx
