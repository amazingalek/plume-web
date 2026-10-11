set(CMAKE_CXX_STANDARD 17)

# Enable SDL Vulkan integration (Linux only)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(PLUME_SDL_VULKAN_ENABLED ON CACHE BOOL "Enable SDL Vulkan integration" FORCE)
endif()

if(EMSCRIPTEN)
    # SDL2 from Emscripten's port, the page from its default shell (a canvas with id "canvas"), the WebGPU device
    # from plume_webgpu_device.js, and ASYNCIFY so the examples' loops can yield to the browser every frame.
    # Configured with -DCMAKE_CXX_FLAGS=-pthread -DCMAKE_C_FLAGS=-pthread instead, an example draws from a thread of
    # its own and the page's main thread runs the device for it (plume_device_thread.h); that needs no ASYNCIFY, and
    # the page needs cross-origin isolation (Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers).
    add_compile_options(-sUSE_SDL=2)
    if(NOT CMAKE_CXX_FLAGS MATCHES "-pthread")
        add_link_options(-sASYNCIFY)
    endif()
    add_link_options(-sUSE_SDL=2 -sALLOW_MEMORY_GROWTH
        "-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$addRunDependency,$removeRunDependency"
        "--pre-js=${CMAKE_SOURCE_DIR}/plume_webgpu_device.js")
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    set(SDL2_INCLUDE_DIRS "")
    set(SDL2_LIBRARIES "")
else()
    # Find or fetch SDL2 (required for examples)
    include(${CMAKE_SOURCE_DIR}/examples/cmake/modules/PlumeSDL2.cmake)
    plume_find_sdl2()
endif()

# Initialize shader compilation
include(${CMAKE_SOURCE_DIR}/examples/cmake/PlumeShaders.cmake)
plume_shaders_init()

# Add example subdirectories
add_subdirectory(triangle)
add_subdirectory(cube)
