# 设置 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(videoQuality)

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 打印 OpenCV 信息
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# 添加目标程序，并链接 OpenCV 库
add_executable(freeze freeze.cpp)
target_link_libraries(freeze ${OpenCV_LIBS})

add_executable(pianse pianse.cpp)
target_link_libraries(pianse ${OpenCV_LIBS})

add_executable(zhedang zhedang.cpp)
target_link_libraries(zhedang ${OpenCV_LIBS})