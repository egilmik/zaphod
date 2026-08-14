cmake_minimum_required(VERSION 3.25)
project(Zaphod VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)




# Flags for GCC/Clang
set(CMAKE_CXX_FLAGS "${cmake_cxx_flags_relwithdebinfo} -march=x86-64-v3 -mtune=generic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64-v3 -mtune=generic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -march=x86-64-v3 -mtune=generic" )

set(NETWORK_FILE_NAME "768-256x2-1_1308206_fe8b690_10k_wdl_0.75.bin")
set(NETWORK_FILE "${CMAKE_CURRENT_SOURCE_DIR}/nnue/${NETWORK_FILE_NAME}")
file(DOWNLOAD "https://raw.githubusercontent.com/egilmik/Zaphod-nnue/main/${NETWORK_FILE_NAME}"
     "${NETWORK_FILE}"
     EXPECTED_HASH SHA256=3bfa95e1f111155646c3c84882a3143d9bfe88a739f3380675406fc8797e63c5)


file(GLOB src CONFIGURE_DEPENDS "*.h" "*.cpp")

add_executable(Zaphod src/board.cpp src/tools.h src/move.h src/bitboard.h src/board.h src/main.cpp "src/zobrist.h" src/material.h src/movegenerator.cpp src/movegenerator.h src/params.h "src/perft/perft.h"  "src/perft/perfttest.cpp" "src/perft/perfttest.h" src/search.h src/search.cpp src/uci.cpp src/uci.h  "src/ttable.h"    "src/nnueq.h" "src/nnueq.cpp" "src/tools/openingbook.h" "src/tools/openingbook.cpp" "src/tools/fentools.h" "src/ttable.cpp"  "src/see.h" "src/history.h")
add_executable(GenerateData src/board.cpp src/tools.h src/move.h src/bitboard.h src/board.h "src/zobrist.h" src/material.h src/movegenerator.cpp src/movegenerator.h src/params.h "src/perft/perft.h"  "src/perft/perfttest.cpp" "src/perft/perfttest.h" src/search.h src/search.cpp src/uci.cpp src/uci.h  "src/ttable.h" "src/generatedata.cpp"    "src/nnueq.h" "src/nnueq.cpp" "src/tools/openingbook.h" "src/tools/openingbook.cpp" "src/tools/fentools.h" "src/ttable.cpp"  "src/see.h" "src/history.h")

target_include_directories(Zaphod PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
target_compile_definitions(Zaphod PRIVATE EVALFILE="${NETWORK_FILE}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(SAN_COMPILE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -O1 -g -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_DEBUG)
set(SAN_LINK -fsanitize=address -fsanitize=undefined)
#target_compile_options(Benchmark PRIVATE ${SAN_COMPILE})
#target_link_options(Benchmark PRIVATE ${SAN_LINK})