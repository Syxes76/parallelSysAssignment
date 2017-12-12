cmake_minimum_required(VERSION 3.7)
project(sobel_filter-code CXX)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(MPI REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set (CMAKE_CXX_FLAGS "-std=c++14")
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin )

add_subdirectory(platforms_devices)
add_subdirectory(simple_opencl)
add_subdirectory(opencl_events)
add_subdirectory(workspace_query)
add_subdirectory(comparator)
add_subdirectory(sobel_filter)
add_subdirectory(sobel_filter_mt)
add_subdirectory(sobel_filter_mpi)
add_subdirectory(sobel_filter_opencl)

if(MPI_COMPILE_FLAGS)
  set_target_properties(sobel_filter_mpi PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -std=c++14")
endif()
if(MPI_LINK_FLAGS)
  set_target_properties(sobel_filter_mpi PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()