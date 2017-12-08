set(sources sobel_filter_opencl.cpp)
set(my_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(sobel_filter_opencl ${sources})

target_include_directories(sobel_filter_opencl PRIVATE 
    ${OpenCV_LIBS}
    ${OpenCL_INCLUDE_DIRS}
    ${my_include_dirs})

target_link_libraries(sobel_filter_opencl ${OpenCL_LIBRARIES} ${OpenCV_LIBS})