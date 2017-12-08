set(sources sobel_filter_mt.cpp)
set (CMAKE_CXX_FLAGS "-pthread -std=c++14")

add_executable(sobel_filter_mt ${sources})

target_include_directories(sobel_filter_mt PRIVATE 
    ${OpenCV_LIBS})

target_link_libraries(sobel_filter_mt ${OpenCV_LIBS})