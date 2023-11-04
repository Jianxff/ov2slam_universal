cmake_minimum_required(VERSION 2.8.3)
project(ov2slam)

#######
# NOTE!
#######
add_definitions(-DOPENCV_CONTRIB)
add_definitions(-DIBOW_LCD)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")


## System dependencies are found with CMake's conventions

# OpenCV
find_package(OpenCV 3 REQUIRED)
#Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

# Ceres
set(Ceres_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/install/lib/cmake/Ceres)
find_package(Ceres REQUIRED)

#Sophus
# First search for a custom build of Sophus within this project
set(Sophus_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/install/share/sophus/cmake) 
find_package(Sophus REQUIRED)

find_package(Pangolin REQUIRED)

###########
## Build ##
###########

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/camera_calibration.cpp
  src/ceres_parametrization.cpp
  src/estimator.cpp
  src/feature_extractor.cpp
  src/feature_tracker.cpp
  src/frame.cpp
  src/loop_closer.cpp
  src/map_manager.cpp
  src/map_point.cpp
  src/mapper.cpp
  src/multi_view_geometry.cpp
  src/optimizer.cpp
  src/ov2slam.cpp
  src/slam_params.cpp
  src/visual_front_end.cpp
)

# Add mandatory dependencies
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/ceres_parametrization
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
      Eigen3::Eigen
      Sophus::Sophus
    PRIVATE
      ${OpenCV_LIBS}
      Ceres::ceres
)


target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/Thirdparty/obindex2/lib/include/
    ${PROJECT_SOURCE_DIR}/Thirdparty/ibow_lcd/include/
)

target_link_libraries(
  ${PROJECT_NAME}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/Thirdparty/install/lib/liblcdetector.so
      ${PROJECT_SOURCE_DIR}/Thirdparty/install/lib/libobindex2.so
)



## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

add_executable(mono_euroc 
  example/mono_euroc.cpp 
  src/visualize.cpp
)

## Specify libraries to link a library or executable target against
target_link_libraries(
    mono_euroc
    PRIVATE
      ${PROJECT_NAME}
      ${Pangolin_LIBRARIES}
)
