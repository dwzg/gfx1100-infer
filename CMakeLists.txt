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
  src/core/chat.cpp
  src/core/gguf.cpp
  src/core/inkling.cpp
  src/core/json.cpp
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
foreach(fmt Q4_K Q5_K Q6_K Q3_K IQ4_XS IQ4_NL Q4_0 Q8_0 NVFP4 IQ3_S IQ3_XXS IQ2_S IQ2_XS IQ2_XXS)
  string(TOLOWER ${fmt} fmt_lower)
  gfx_shader(matvec_${fmt_lower} matvec_q.comp FMT_${fmt})
  gfx_shader(matmat_${fmt_lower} matvec_q.comp FMT_${fmt} BATCH)
  gfx_shader(wmma_${fmt_lower} matvec_q.comp FMT_${fmt} WMMA)
  gfx_shader(wmmaemu_${fmt_lower} matvec_q.comp FMT_${fmt} WMMA WMMA_EMU)
  gfx_shader(expert_${fmt_lower} matvec_q.comp FMT_${fmt} EXPERT)
endforeach()
gfx_shader(blocksum blocksum.comp)
gfx_shader(memread memread.comp)
gfx_shader(norm norm.comp)
gfx_shader(swiglu swiglu.comp)
gfx_shader(attn_prep attn_prep.comp)
gfx_shader(attn_prep_kv8 attn_prep.comp KV8)
gfx_shader(attn_score attn_score.comp)
gfx_shader(attn_score_kv8 attn_score.comp KV8)
gfx_shader(attn_soft attn_soft.comp)
gfx_shader(attn_vsum attn_vsum.comp)
gfx_shader(attn_vsum_kv8 attn_vsum.comp KV8)
gfx_shader(attn_combine attn_combine.comp)
gfx_shader(dn_prep dn_prep.comp)
gfx_shader(dn_update dn_update.comp)
gfx_shader(dn_hist dn_hist.comp)
gfx_shader(matmat_f16 matmat_f16.comp)
gfx_shader(to_f16 to_f16.comp)
# Inkling 2
gfx_shader(matvec_f32 matvec_f32.comp)
gfx_shader(ink_mix ink_mix.comp)
gfx_shader(ink_ctl_in ink_ctl.comp)
gfx_shader(ink_ctl_out ink_ctl.comp OUT)
gfx_shader(ink_conv ink_conv.comp)
gfx_shader(ink_table ink_table.comp)
gfx_shader(ink_route ink_route.comp)
gfx_shader(ink_situ ink_situ.comp)
gfx_shader(ink_esum ink_esum.comp)
gfx_shader(ink_prep ink_prep.comp)
gfx_shader(ink_prep_kv8 ink_prep.comp KV8)
gfx_shader(ink_score ink_score.comp)
gfx_shader(ink_score_kv8 ink_score.comp KV8)
gfx_shader(ink_wscore ink_score.comp RING)
gfx_shader(ink_wscore_kv8 ink_score.comp KV8 RING)
gfx_shader(ink_soft attn_soft.comp SINK)
gfx_shader(ink_wsoft attn_soft.comp SINK RING)
gfx_shader(ink_wvsum attn_vsum.comp RING)
gfx_shader(ink_wvsum_kv8 attn_vsum.comp KV8 RING)
gfx_shaders_finish()

add_library(gfxgpu STATIC
  src/engine/gpu_model.cpp src/engine/gpu_inkling.cpp src/engine/context.cpp
  src/engine/pack.cpp
  src/vk/vk.cpp
  src/vk/shaders.cpp)
target_link_libraries(gfxgpu PUBLIC gfxcore gfxshaders Vulkan::Vulkan)

find_package(Threads REQUIRED)
add_library(gfxserver STATIC
  src/server/prompt.cpp
  src/server/reply.cpp
  src/server/server.cpp)
target_link_libraries(gfxserver PUBLIC gfxgpu Threads::Threads)

add_executable(gfxi src/tools/main.cpp)
target_link_libraries(gfxi PRIVATE gfxserver)
