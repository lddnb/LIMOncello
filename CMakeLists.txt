cmake_minimum_required(VERSION 3.22)
project(limoncello VERSION 0.0.1)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread -fexceptions -O3 -lm -lblas -llapack")

list(APPEND CMAKE_PREFIX_PATH "../../slam_common/install" "../../slam_core/install")

find_package(Boost REQUIRED COMPONENTS system)
find_package(TBB REQUIRED COMPONENTS tbb)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io)
find_package(slam_common REQUIRED)
find_package(slam_core REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/external/manif)

add_executable(${PROJECT_NAME}
  src/main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  include
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ../../slam_adapter/include
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    manif
    slam_common::slam_common
    slam_core::slam_core
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    TBB::tbb
    spdlog::spdlog
    yaml-cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  PCL_NO_PRECOMPILE
)
