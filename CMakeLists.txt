cmake_minimum_required(VERSION 3.20.0)
project(Zaphod VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)



if (MSVC)
    # Flags for MSVC
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /GL /arch:AVX2 /fp:fast /GR- /EHs-")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  /Ob2 /GL /arch:AVX2 /fp:fast /GR- /EHs-")
else()
    # Flags for GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=x86-64-v3 -mavx2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -march=x86-64-v3 -mavx2")
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


file(GLOB src CONFIGURE_DEPENDS "*.h" "*.cpp")

add_executable(Zaphod src/board.cpp src/tools.h src/move.h src/bitboard.h src/board.h src/main.cpp src/transpositiontable.h src/material.h src/movegenerator.cpp src/movegenerator.h "src/perft/perft.h"  "src/perft/perfttest.cpp" "src/perft/perfttest.h" src/search.h src/search.cpp src/uci.cpp src/uci.h  "src/ttable.h"    "src/nnueq.h" "src/nnueq.cpp" "src/tools/openingbook.h" "src/tools/openingbook.cpp" "src/tools/fentools.h" "src/ttable.cpp")
add_executable(Benchmark src/board.cpp src/tools.h src/move.h src/bitboard.h src/board.h src/benchmark.cpp src/transpositiontable.h src/material.h src/movegenerator.cpp src/movegenerator.h "src/perft/perft.h"  "src/perft/perfttest.cpp" "src/perft/perfttest.h" src/search.h src/search.cpp  "src/ttable.h"     "src/nnueq.h" "src/nnueq.cpp" "src/tools/openingbook.h" "src/tools/openingbook.cpp" "src/tools/fentools.h" "src/ttable.cpp")
add_executable(GenerateData src/board.cpp src/tools.h src/move.h src/bitboard.h src/board.h src/transpositiontable.h src/material.h src/movegenerator.cpp src/movegenerator.h "src/perft/perft.h"  "src/perft/perfttest.cpp" "src/perft/perfttest.h" src/search.h src/search.cpp src/uci.cpp src/uci.h  "src/ttable.h" "src/generatedata.cpp"    "src/nnueq.h" "src/nnueq.cpp" "src/tools/openingbook.h" "src/tools/openingbook.cpp" "src/tools/fentools.h" "src/ttable.cpp")

target_include_directories(Zaphod PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(SAN_COMPILE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -O1 -g -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_DEBUG)
set(SAN_LINK -fsanitize=address -fsanitize=undefined)
#target_compile_options(Benchmark PRIVATE ${SAN_COMPILE})
#target_link_options(Benchmark PRIVATE ${SAN_LINK})
### Testing ###

enable_testing()

add_executable(
  zaphod_test
  test/testboard.cpp test/testttable.cpp test/testperft.cpp src/move.h src/bitboard.h test/testtranspositiontable.cpp test/testsearch.cpp test/testuci.cpp src/uci.cpp src/tools.h src/uci.h src/board.h src/board.cpp src/search.h src/search.cpp "src/perft/perft.h" src/perft/perfttest.cpp src/perft/perfttest.h src/movegenerator.cpp src/transpositiontable.h src/movegenerator.h
 "test/testtools.cpp" "test/testmove.cpp" "test/testmaterial.cpp"   "src/ttable.h"     "src/nnueq.h" "src/nnueq.cpp" "src/tools/openingbook.h" "src/tools/openingbook.cpp" "src/tools/fentools.h" "src/ttable.cpp")


target_link_libraries(
  zaphod_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(zaphod_test)

