cmake_minimum_required (VERSION 3.8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project("ml" LANGUAGES C CUDA)
add_subdirectory(src)
