cmake_minimum_required(VERSION 3.20)
project(gfx1100-infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
option(GFX_NATIVE "Optimise the CPU code for this machine" ON)
if(GFX_NATIVE)
  add_compile_options(-march=native)
endif()
find_package(Threads REQUIRED)

add_library(gfxcore STATIC
  src/core/gguf.cpp
  src/core/model.cpp
  src/core/quant.cpp
  src/core/ref.cpp
  src/core/regex.cpp
  src/core/sampler.cpp
  src/core/tokenizer.cpp
  src/core/unicode.cpp)
target_include_directories(gfxcore PUBLIC src)
target_link_libraries(gfxcore PUBLIC Threads::Threads)

find_package(Vulkan REQUIRED)
include(cmake/shaders.cmake)
gfx_shader(matvec_f16 matvec_f16.comp)
foreach(fmt Q4_K Q5_K Q6_K Q3_K IQ4_XS IQ4_NL Q4_0 Q8_0 NVFP4)
  string(TOLOWER ${fmt} fmt_lower)
  gfx_shader(matvec_${fmt_lower} matvec_q.comp FMT_${fmt})
endforeach()
gfx_shader(blocksum blocksum.comp)
gfx_shader(norm norm.comp)
gfx_shader(swiglu swiglu.comp)
gfx_shader(attn_prep attn_prep.comp)
gfx_shader(attn attn.comp)
gfx_shader(dn_prep dn_prep.comp)
gfx_shader(dn_update dn_update.comp)
gfx_shaders_finish()

add_library(gfxgpu STATIC
  src/engine/gpu_model.cpp
  src/engine/pack.cpp
  src/vk/vk.cpp
  src/vk/shaders.cpp)
target_link_libraries(gfxgpu PUBLIC gfxcore gfxshaders Vulkan::Vulkan)

add_executable(gfxi src/tools/main.cpp)
target_link_libraries(gfxi PRIVATE gfxgpu)
