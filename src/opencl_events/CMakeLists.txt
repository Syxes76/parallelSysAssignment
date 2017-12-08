set(sources opencl_events.cpp)
set(my_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(opencl_events ${sources})

target_include_directories(opencl_events PRIVATE 
    ${OpenCL_INCLUDE_DIRS}
    ${my_include_dirs})

target_link_libraries(opencl_events ${OpenCL_LIBRARIES})
