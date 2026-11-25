# Compile GLSL compute kernels to SPIR-V and embed them into the binary.
find_program(GLSLANG glslangValidator REQUIRED)
set(GFX_SHADER_DIR ${CMAKE_SOURCE_DIR}/src/shaders)
set(GFX_SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${GFX_SHADER_OUT})
file(GLOB GFX_SHADER_INCLUDES ${GFX_SHADER_DIR}/*.glsl)
set(GFX_SHADER_HEADERS "")
set(GFX_SHADER_NAMES "")

# gfx_shader(<kernel name> <source file> [defines...])
function(gfx_shader name source)
  set(defs "")
  foreach(d ${ARGN})
    list(APPEND defs "-D${d}")
  endforeach()
  set(out ${GFX_SHADER_OUT}/${name}.h)
  add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} -V --quiet --target-env vulkan1.3 -I${GFX_SHADER_DIR} ${defs} --vn spv_${name} -o ${out} ${GFX_SHADER_DIR}/${source}
    DEPENDS ${GFX_SHADER_DIR}/${source} ${GFX_SHADER_INCLUDES}
    COMMENT "kernel ${name}" VERBATIM)
  set(GFX_SHADER_HEADERS ${GFX_SHADER_HEADERS} ${out} PARENT_SCOPE)
  set(GFX_SHADER_NAMES ${GFX_SHADER_NAMES} ${name} PARENT_SCOPE)
endfunction()

# Call after all gfx_shader() calls: writes the registry source and defines the gfxshaders target.
function(gfx_shaders_finish)
  set(src "#include <cstdint>\n#include <cstddef>\n#include \"vk/vk.h\"\n")
  set(table "")
  foreach(n ${GFX_SHADER_NAMES})
    string(APPEND src "#include \"shaders/${n}.h\"\n")
    string(APPEND table "    {\"${n}\", spv_${n}, sizeof(spv_${n})},\n")
  endforeach()
  string(APPEND src "namespace gfx {\nconst std::vector<ShaderBlob> &all_shaders() {\n  static const std::vector<ShaderBlob> v = {\n${table}  };\n  return v;\n}\n}\n")
  file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/shader_registry.cpp CONTENT "${src}")
  add_library(gfxshaders STATIC ${CMAKE_BINARY_DIR}/shader_registry.cpp ${GFX_SHADER_HEADERS})
  target_include_directories(gfxshaders PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
endfunction()
