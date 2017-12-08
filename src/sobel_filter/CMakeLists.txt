set(sources sobel_filter.cpp)

add_executable(sobel_filter ${sources})

target_include_directories(sobel_filter PRIVATE 
    ${OpenCV_LIBS})

target_link_libraries(sobel_filter ${OpenCV_LIBS})