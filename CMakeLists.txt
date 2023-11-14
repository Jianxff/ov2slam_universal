cmake_minimum_required(VERSION 3.0)
project(ov2slam)

##########################
# cmake flags
##########################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")


##########################
# project options
##########################
option(USE_BRIEF "Use brief descriptor" ON)
option(USE_LOOP_CLOSER "Use Loop-Closer" ON)

if(USE_LOOP_CLOSER)
  add_definitions(-DIBOW_LCD)
endif(USE_LOOP_CLOSER)

if(USE_BRIEF)
  add_definitions(-DOPENCV_CONTRIB)
endif(USE_BRIEF)


##########################
# find package
##########################
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)


##########################
# pybind11
##########################
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/pybind11)


##########################
# build pybind module
##########################
pybind11_add_module(${PROJECT_NAME}
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
  src/visualize.cpp
  src/wrapper_py.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/ceres_parametrization
    ${PROJECT_SOURCE_DIR}/Thirdparty/pybind11/include/
    /usr/include/python3.8
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
      Eigen3::Eigen
      Sophus::Sophus
    PRIVATE
      ${OpenCV_LIBS}
      Ceres::ceres
      ${Pangolin_LIBRARIES}
)


##########################
# use loop closer
##########################
if(USE_LOOP_CLOSER)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
      ${PROJECT_SOURCE_DIR}/Thirdparty/obindex2/lib/include/
      ${PROJECT_SOURCE_DIR}/Thirdparty/ibow_lcd/include/
  )

  target_link_libraries(
    ${PROJECT_NAME}
      PRIVATE
        ${PROJECT_SOURCE_DIR}/Thirdparty/ibow_lcd/build/liblcdetector.so
  )
endif(USE_LOOP_CLOSER)