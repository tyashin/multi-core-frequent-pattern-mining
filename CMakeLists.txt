cmake_minimum_required(VERSION 3.20)
project(fpm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FPM_BUILD_TESTS "Build GoogleTest suite" ON)
option(FPM_BUILD_PYTHON "Build the pybind11 Python module" OFF)
option(FPM_BUILD_SHARED "Build libfpm.so (C ABI)" ON)
# Usage: -DFPM_SANITIZE="address,undefined"
set(FPM_SANITIZE "" CACHE STRING "Sanitizers to enable on fpm targets")

find_package(OpenMP REQUIRED)

add_library(fpm_core STATIC
  src/bitset.cpp
  src/mine.cpp
  src/rules.cpp
  src/witness.cpp
  src/abi.cpp
)
target_include_directories(fpm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fpm_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(fpm_core PRIVATE -Wall -Wextra -Werror)

if(FPM_SANITIZE)
  target_compile_options(fpm_core PUBLIC -fsanitize=${FPM_SANITIZE} -fno-omit-frame-pointer)
  target_link_options(fpm_core PUBLIC -fsanitize=${FPM_SANITIZE})
endif()

if(FPM_BUILD_SHARED)
  add_library(fpm_shared SHARED src/abi.cpp)
  target_link_libraries(fpm_shared PRIVATE fpm_core)
  set_target_properties(fpm_shared PROPERTIES OUTPUT_NAME fpm)
  target_compile_options(fpm_shared PRIVATE -Wall -Wextra -Werror)
endif()

if(FPM_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(fpm src/pymodule.cpp)
  target_link_libraries(fpm PRIVATE fpm_core)
  install(TARGETS fpm LIBRARY DESTINATION .)
endif()

if(FPM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(fpm_tests
    tests/test_bitset.cpp
    tests/test_mine.cpp
    tests/test_rules.cpp
    tests/test_abi.cpp
  )
  target_link_libraries(fpm_tests PRIVATE fpm_core GTest::gtest_main)
  target_compile_options(fpm_tests PRIVATE -Wall -Wextra -Werror)
  include(GoogleTest)
  gtest_discover_tests(fpm_tests)
endif()
