cmake_minimum_required(VERSION 3.7)
project(opencl-code CXX)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set (CMAKE_CXX_FLAGS "-std=c++14")
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin )

add_subdirectory(platforms_devices)
add_subdirectory(simple_opencl)
add_subdirectory(opencl_events)
add_subdirectory(workspace_query)
add_subdirectory(sobel_filter)
