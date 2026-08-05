cmake_minimum_required(VERSION 3.25.0)
project(Zaphod VERSION 2.0.0)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

# A plain configure-and-build must reproduce the release binary.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "No build type specified, defaulting to Release")
endif()

# src/nnueq.cpp embeds the network with #embed, which needs GCC >= 15 or Clang >= 19.
option(ZAPHOD_SKIP_COMPILER_CHECK "Skip the #embed compiler-support check" OFF)
if(NOT ZAPHOD_SKIP_COMPILER_CHECK)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15)
        message(FATAL_ERROR "Zaphod uses #embed and needs GCC >= 15 (found ${CMAKE_CXX_COMPILER_VERSION})")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
        message(FATAL_ERROR "Zaphod uses #embed and needs Clang >= 19 (found ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
endif()

set(EVALFILE "nnue/768-256x2-1_29112025_556M_10k_wdl_0.75.bin" CACHE STRING "Path to the evaluation network file")

find_package(Threads REQUIRED)

set(ZAPHOD_SOURCES
    src/board.cpp
    src/movegenerator.cpp
    src/search.cpp
    src/uci.cpp
    src/ttable.cpp
    src/nnueq.cpp
    src/perft/perfttest.cpp
    src/tools/openingbook.cpp
)

add_executable(Zaphod ${ZAPHOD_SOURCES} src/main.cpp)
add_executable(GenerateData ${ZAPHOD_SOURCES} src/generatedata.cpp)

foreach(tgt Zaphod GenerateData)
    # The NNUE path requires AVX2; x86-64-v3 is the release baseline.
    if(MSVC)
        target_compile_options(${tgt} PRIVATE /arch:AVX2)
    else()
        target_compile_options(${tgt} PRIVATE -march=x86-64-v3 -mtune=generic)
    endif()
    target_compile_definitions(${tgt} PRIVATE
        EVALFILE="${EVALFILE}"
        ZAPHOD_VERSION="${PROJECT_VERSION}")
    target_link_libraries(${tgt} PRIVATE Threads::Threads)
endforeach()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set_target_properties(Zaphod GenerateData PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(STATUS "IPO/LTO not supported: ${ipo_error}")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Sanitizer flags for local debugging, e.g.:
#   target_compile_options(Zaphod PRIVATE ${SAN_COMPILE})
#   target_link_options(Zaphod PRIVATE ${SAN_LINK})
set(SAN_COMPILE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -O1 -g -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_DEBUG)
set(SAN_LINK -fsanitize=address -fsanitize=undefined)
