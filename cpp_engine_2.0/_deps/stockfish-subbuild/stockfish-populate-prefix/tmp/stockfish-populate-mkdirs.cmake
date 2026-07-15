# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-src")
  file(MAKE_DIRECTORY "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-src")
endif()
file(MAKE_DIRECTORY
  "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-build"
  "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-subbuild/stockfish-populate-prefix"
  "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-subbuild/stockfish-populate-prefix/tmp"
  "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-subbuild/stockfish-populate-prefix/src/stockfish-populate-stamp"
  "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-subbuild/stockfish-populate-prefix/src"
  "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-subbuild/stockfish-populate-prefix/src/stockfish-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-subbuild/stockfish-populate-prefix/src/stockfish-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/sid/Documents/GitHub/G-ForceZero/cpp_engine_2.0/_deps/stockfish-subbuild/stockfish-populate-prefix/src/stockfish-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
