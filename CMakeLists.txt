cmake_minimum_required(VERSION 3.0.0)
project(Zaphod VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


file(GLOB src CONFIGURE_DEPENDS "*.h" "*.cpp")

add_executable(Zaphod src/board.cpp src/board.h src/main.cpp src/movegenerator.cpp src/movegenerator.h src/perft.h src/test.cpp src/test.h)

target_include_directories(Zaphod PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

### Testing ###

enable_testing()

add_executable(
  zaphod_test
  test/testboard.cpp test/testperft.cpp src/board.h src/board.cpp src/perft.h src/movegenerator.cpp src/movegenerator.h
)
target_link_libraries(
  zaphod_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(zaphod_test)

