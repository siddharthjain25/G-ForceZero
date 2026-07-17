cmake_minimum_required(VERSION 3.10)
project(ChessEngineCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

# Output binaries directly to cpp_engine_2.0 directory instead of build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Maximum performance flags: generic x86-64 v2 (SSE4.1, POPCNT), LTO
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.1 -mpopcnt")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto")

add_executable(gforce_engine src/gforce_engine.cpp src/tbprobe.cpp)
target_link_libraries(gforce_engine pthread)

add_executable(pgn_to_epd src/pgn_to_epd.cpp)
target_link_libraries(pgn_to_epd pthread)
