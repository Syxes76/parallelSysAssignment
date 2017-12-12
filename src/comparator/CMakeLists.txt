set(sources comparator.cpp)

add_executable(comparator ${sources})

target_include_directories(comparator PRIVATE 
    ${OpenCV_LIBS})

target_link_libraries(comparator ${OpenCV_LIBS})