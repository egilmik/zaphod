cmake_minimum_required(VERSION 3.25)
project(Zaphod VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language / global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single config generators (Ninja, Unix Makefiles) have no build type by
# default, which silently produces an unoptimized engine.
get_property(_zaphod_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _zaphod_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(ZAPHOD_ARCH "x86-64-v3" CACHE STRING
    "-march= value used by GCC/Clang (GNU driver). x86-64-v3 implies AVX2/BMI2.")
set(ZAPHOD_ARCH_MSVC "/arch:AVX2" CACHE STRING
    "Architecture flag used by MSVC style drivers (cl, clang-cl).")
option(ZAPHOD_SANITIZE "Build with address + undefined behaviour sanitizers" OFF)

set(ZAPHOD_NETWORK_NAME "768-256x2-1_1308206_fe8b690_10k_wdl_0.75.bin" CACHE STRING
    "File name of the NNUE network to embed")
set(ZAPHOD_NETWORK_SHA256 "3bfa95e1f111155646c3c84882a3143d9bfe88a739f3380675406fc8797e63c5"
    CACHE STRING "Expected SHA256 of ZAPHOD_NETWORK_NAME")
set(ZAPHOD_NETWORK_URL "https://raw.githubusercontent.com/egilmik/Zaphod-nnue/main"
    CACHE STRING "Base URL the network is downloaded from")
set(ZAPHOD_NETWORK_FILE "" CACHE FILEPATH
    "Network file to embed. Skips the download when set")

# ---------------------------------------------------------------------------
# Compiler configuration
#
# The flags are attached to targets instead of CMAKE_CXX_FLAGS so that a
# clang-cl / MSVC build does not get handed GNU driver flags and vice versa.
# ---------------------------------------------------------------------------
add_library(zaphod_flags INTERFACE)

if(MSVC)
    # cl.exe and clang-cl.exe. MSVC is also true for clang-cl, which is the
    # combination Visual Studio uses through the ClangCL toolset.
    target_compile_options(zaphod_flags INTERFACE
        ${ZAPHOD_ARCH_MSVC}
        /EHsc
        /utf-8
        $<$<CXX_COMPILER_ID:MSVC>:/permissive->
    )
    target_compile_definitions(zaphod_flags INTERFACE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
else()
    # GCC and Clang with the GNU driver
    target_compile_options(zaphod_flags INTERFACE
        -march=${ZAPHOD_ARCH}
        -mtune=generic
    )
endif()

if(ZAPHOD_SANITIZE)
    if(MSVC)
        target_compile_options(zaphod_flags INTERFACE /fsanitize=address /Zi)
    else()
        target_compile_options(zaphod_flags INTERFACE
            -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -g)
        target_link_options(zaphod_flags INTERFACE
            -fsanitize=address -fsanitize=undefined)
    endif()
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Engine core - everything shared between the executables
# ---------------------------------------------------------------------------
add_library(zaphod_core STATIC
    src/bitboard.h
    src/board.cpp
    src/board.h
    src/history.h
    src/material.h
    src/move.h
    src/movegenerator.cpp
    src/movegenerator.h
    src/nnueq.cpp
    src/nnueq.h
    src/params.h
    src/perft/perft.h
    src/perft/perfttest.cpp
    src/perft/perfttest.h
    src/search.cpp
    src/search.h
    src/see.h
    src/tools.h
    src/tools/fentools.h
    src/tools/openingbook.cpp
    src/tools/openingbook.h
    src/ttable.cpp
    src/ttable.h
    src/uci.cpp
    src/uci.h
    src/zobrist.h
)

target_include_directories(zaphod_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(zaphod_core PUBLIC zaphod_flags Threads::Threads)

# ---------------------------------------------------------------------------
# NNUE network
#
# The net is fetched into nnue/ and baked into the binary by the #embed in
# src/nnueq.cpp. file(DOWNLOAD) skips the transfer when the file is already
# there with the expected hash, and fails the configure step when what it
# downloaded has any other hash.
# ---------------------------------------------------------------------------
if(NOT ZAPHOD_NETWORK_FILE)
    set(ZAPHOD_NETWORK_FILE "${CMAKE_CURRENT_SOURCE_DIR}/nnue/${ZAPHOD_NETWORK_NAME}")
    file(DOWNLOAD "${ZAPHOD_NETWORK_URL}/${ZAPHOD_NETWORK_NAME}" "${ZAPHOD_NETWORK_FILE}"
         EXPECTED_HASH SHA256=${ZAPHOD_NETWORK_SHA256}
         SHOW_PROGRESS)
endif()

# #embed is the only way the network gets linked in, so say so up front
# instead of leaving the compiler to complain about an unknown directive.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_QUIET TRUE)
check_cxx_source_compiles("
#if !defined(__has_embed) || __has_embed(__FILE__) != 1
#  error \"no usable #embed\"
#endif
int main() { return 0; }
" ZAPHOD_COMPILER_HAS_EMBED)
if(NOT ZAPHOD_COMPILER_HAS_EMBED)
    message(FATAL_ERROR
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} cannot compile #embed, "
        "which Zaphod needs to link the network in.\n"
        "Use GCC 15 or newer, or Clang 19 or newer (clang-cl included).")
endif()

# Forward slashes work in string literals on every platform, backslashes do
# not survive the trip through the preprocessor.
file(TO_CMAKE_PATH "${ZAPHOD_NETWORK_FILE}" ZAPHOD_NNUE_FILE_PATH)
file(SIZE "${ZAPHOD_NETWORK_FILE}" ZAPHOD_NNUE_FILE_BYTES)
file(CONFIGURE
    OUTPUT "${CMAKE_BINARY_DIR}/generated/zaphod_nnue_embed.h"
    CONTENT [[
// Generated by CMakeLists.txt - do not edit.
#ifndef ZAPHOD_NNUE_EMBED_H
#define ZAPHOD_NNUE_EMBED_H
#define ZAPHOD_NNUE_FILE "@ZAPHOD_NNUE_FILE_PATH@"
#define ZAPHOD_NNUE_FILE_BYTES @ZAPHOD_NNUE_FILE_BYTES@
#endif
]]
    @ONLY)

message(STATUS "Zaphod NNUE: embedding ${ZAPHOD_NETWORK_FILE} (${ZAPHOD_NNUE_FILE_BYTES} bytes)")
target_include_directories(zaphod_core PRIVATE "${CMAKE_BINARY_DIR}/generated")

# #embed is invisible to the dependency scanner, so nnueq.cpp has to be told
# about the network explicitly or a new net never gets picked up.
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/nnueq.cpp"
             APPEND PROPERTY OBJECT_DEPENDS "${ZAPHOD_NETWORK_FILE}")

# Clang reports #embed as a C23 extension in C++ mode.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(zaphod_core PRIVATE -Wno-c23-extensions)
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(Zaphod src/main.cpp)
target_link_libraries(Zaphod PRIVATE zaphod_core)

add_executable(GenerateData src/generatedata.cpp)
target_link_libraries(GenerateData PRIVATE zaphod_core)

# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
install(TARGETS Zaphod RUNTIME DESTINATION bin)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
