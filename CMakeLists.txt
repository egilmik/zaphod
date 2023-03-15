cmake_minimum_required(VERSION 3.0.0)
project(Chess VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

include(CTest)
enable_testing()



add_executable(Chess board.cpp board.h main.cpp movegenerator.cpp movegenerator.h)

target_include_directories(Chess PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
