cmake_minimum_required(VERSION 3.10)
project(ChessEngineCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

# Maximum performance flags: native arch, LTO, fast-math
# Explicit AVX2/FMA/BMI2/POPCNT since confirmed on target CPU (AMD Ryzen 5 3550H)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mbmi2 -mpopcnt")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto -fno-rtti")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto")

add_executable(nnue_engine src/nnue_engine.cpp src/nnue.cpp src/tbprobe.cpp)
target_link_libraries(nnue_engine pthread)

add_executable(pgn_to_epd src/pgn_to_epd.cpp)
target_link_libraries(pgn_to_epd pthread)
