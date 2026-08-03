# CMakeLists.txt

# 指定需要的 CMake 最低版本
cmake_minimum_required(VERSION 3.16)

# 定义项目名称
project(SamplingTools CXX)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Boost 库，并指定我们需要 program_options 这个组件
find_package(Boost REQUIRED COMPONENTS program_options)

# 将 cpp 目录添加到头文件搜索路径
include_directories(cpp)

# --- 可执行文件 1: 原始的下采样脚本 ---
add_executable(downsampler cpp/downsampler.cpp)
target_link_libraries(downsampler PRIVATE Boost::program_options)

# --- 可执行文件 2: 新的嵌套下采样脚本 ---
add_executable(nested_sampler cpp/create_nested_samples.cpp)
target_link_libraries(nested_sampler PRIVATE Boost::program_options)

add_executable(query_downsample cpp/query_downsample.cpp)
target_link_libraries(query_downsample PRIVATE Boost::program_options)

# 设置输出目录，让所有可执行文件都生成在 build 文件夹中
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})