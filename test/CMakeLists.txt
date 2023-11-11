cmake_minimum_required(VERSION 2.8.3)
project(ov2slam)

##########################
# cmake flags
##########################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")


##########################
# project options
##########################
option(MULTI_THREAD "Use multi-threading" ON)
option(USE_BRIEF "Use brief descriptor" ON)

if(MULTI_THREAD)
  message(STATUS "Multi-threading is on")
  add_definitions(-DMULTI_THREAD)
endif(MULTI_THREAD)

if(USE_BRIEF)
  add_definitions(-DOPENCV_CONTRIB)
endif(USE_BRIEF)


##########################
# find package
##########################
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)


##########################
# build test
##########################
add_executable(test_euroc
  ../src/camera_calibration.cpp
  ../src/ceres_parametrization.cpp
  ../src/estimator.cpp
  ../src/feature_extractor.cpp
  ../src/feature_tracker.cpp
  ../src/frame.cpp
  ../src/map_manager.cpp
  ../src/map_point.cpp
  ../src/mapper.cpp
  ../src/multi_view_geometry.cpp
  ../src/optimizer.cpp
  ../src/ov2slam.cpp
  ../src/slam_params.cpp
  ../src/visual_front_end.cpp
  test_euroc.cpp 
  visualize.cpp
)

### include 
target_include_directories(test_euroc
  PUBLIC
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../include/ceres_parametrization
    ${PROJECT_SOURCE_DIR}
)

### linking
target_link_libraries(test_euroc
  PUBLIC
    Eigen3::Eigen
    Sophus::Sophus
  PRIVATE
    Ceres::ceres
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
)