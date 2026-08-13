cmake_minimum_required(VERSION 3.15)
project(fs2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Once the real sources are dropped into native/fs2d/src/, add them here,
# e.g.:
#   file(GLOB FS2D_SOURCES src/*.cpp)
# and rename fs2d_c_api_template.cpp -> fs2d_c_api.cpp.

add_library(fs2d SHARED
    fs2d_c_api_template.cpp
)

target_include_directories(fs2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(fs2d PROPERTIES
    OUTPUT_NAME "fs2d"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)
