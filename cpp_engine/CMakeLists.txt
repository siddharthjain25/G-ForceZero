cmake_minimum_required(VERSION 3.10)
project(ChessEngineCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

add_executable(tuner tuner.cpp)
target_link_libraries(tuner PRIVATE ${CMAKE_THREAD_LIBS_INIT})

add_executable(classical_engine classical_engine.cpp)
target_link_libraries(classical_engine pthread)

add_executable(nnue_engine nnue_engine.cpp nnue.cpp)
target_link_libraries(nnue_engine pthread)
