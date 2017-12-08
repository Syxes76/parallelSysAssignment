set(sources workspace_query.cpp)
set(my_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(workspace_query ${sources})

target_include_directories(workspace_query PRIVATE 
    ${OpenCL_INCLUDE_DIRS}
    ${my_include_dirs})

target_link_libraries(workspace_query ${OpenCL_LIBRARIES})
