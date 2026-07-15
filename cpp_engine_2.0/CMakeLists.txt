cmake_minimum_required(VERSION 3.10)
project(ChessEngineCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

# Output binaries directly to cpp_engine_2.0 directory instead of build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Maximum performance flags: native, LTO, fast-math
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto")

add_subdirectory(src/sf_bridge)
add_executable(nnue_engine src/nnue_engine.cpp src/nnue.cpp src/tbprobe.cpp)
target_link_libraries(nnue_engine sf_bridge pthread)

add_executable(pgn_to_epd src/pgn_to_epd.cpp)
target_link_libraries(pgn_to_epd pthread)
