# WebGPU backend checks (browser only)

add_executable(plume_webgpu_checks main.cpp)
target_link_libraries(plume_webgpu_checks PRIVATE plume)
target_include_directories(plume_webgpu_checks PRIVATE ${CMAKE_SOURCE_DIR})
set_target_properties(plume_webgpu_checks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
