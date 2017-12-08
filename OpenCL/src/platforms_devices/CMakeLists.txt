set(sources platforms_devices.cpp)
set(my_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(platforms_devices ${sources})

target_include_directories(platforms_devices PRIVATE 
    ${OpenCL_INCLUDE_DIRS}
    ${my_include_dirs})

target_link_libraries(platforms_devices ${OpenCL_LIBRARIES})
