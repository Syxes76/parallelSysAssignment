set(sources simple_opencl.cpp)
set(my_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(simple_opencl ${sources})

target_include_directories(simple_opencl PRIVATE 
    ${OpenCL_INCLUDE_DIRS}
    ${my_include_dirs})

target_link_libraries(simple_opencl ${OpenCL_LIBRARIES})
