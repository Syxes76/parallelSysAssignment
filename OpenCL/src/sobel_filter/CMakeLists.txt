set(sources sobel_filter.cpp)
set(my_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(sobel_filter ${sources})

target_include_directories(sobel_filter PRIVATE 
    ${OpenCV_LIBS}
    ${OpenCL_INCLUDE_DIRS}
    ${my_include_dirs})

target_link_libraries(sobel_filter ${OpenCL_LIBRARIES} ${OpenCV_LIBS})