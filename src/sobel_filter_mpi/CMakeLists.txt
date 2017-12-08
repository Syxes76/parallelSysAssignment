set(sources sobel_filter_mpi.cpp)

include_directories(${MPI_INCLUDE_PATH})

add_executable(sobel_filter_mpi ${sources})

target_include_directories(sobel_filter_mpi PRIVATE 
    ${OpenCV_LIBS}
    ${MPI_LIBRARIES})

target_link_libraries(sobel_filter_mpi ${OpenCV_LIBS} ${MPI_LIBRARIES})