cmake_minimum_required(VERSION 3.20)
project(say CXX)

set(LLAMA_DIR D:/drama3/third_party/llama.cpp CACHE PATH "llama.cpp source tree")
set(LLAMA_BUILD ${LLAMA_DIR}/build CACHE PATH "llama.cpp build tree")

add_executable(say say.cpp)
target_compile_features(say PRIVATE cxx_std_17)
target_include_directories(say PRIVATE ${LLAMA_DIR}/include ${LLAMA_DIR}/ggml/include ${LLAMA_DIR}/tools/mtmd)

if(WIN32)
    # import libraries of an MSVC build
    target_link_libraries(say PRIVATE
        ${LLAMA_BUILD}/tools/mtmd/mtmd.lib
        ${LLAMA_BUILD}/src/llama.lib
        ${LLAMA_BUILD}/ggml/src/ggml-base.lib)

    file(GLOB dlls ${LLAMA_BUILD}/bin/ggml*.dll ${LLAMA_BUILD}/bin/llama.dll ${LLAMA_BUILD}/bin/mtmd.dll ${LLAMA_BUILD}/bin/cublas*.dll)
    add_custom_command(TARGET say POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dlls} $<TARGET_FILE_DIR:say>)
else()
    target_link_directories(say PRIVATE ${LLAMA_BUILD}/bin)
    target_link_libraries(say PRIVATE mtmd llama ggml-base)
    set_target_properties(say PROPERTIES BUILD_RPATH ${LLAMA_BUILD}/bin INSTALL_RPATH "$ORIGIN")
endif()
