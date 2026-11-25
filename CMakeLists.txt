cmake_minimum_required(VERSION 3.20)
project(gfx1100-infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)
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

add_executable(gfxi src/tools/main.cpp)
target_link_libraries(gfxi PRIVATE gfxcore)
