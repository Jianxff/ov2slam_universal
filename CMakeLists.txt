cmake_minimum_required(VERSION 2.8.3)
project(webslam)

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
option(JS_DEBUG "Open debug mode" ON)

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
set(Eigen3_DIR ~/shared/eigen-3.4.0/install_js/share/eigen3/cmake)
set(Sophus_DIR ~/shared/Sophus/install_js/share/sophus/cmake)
set(Ceres_DIR ~/shared/ceres-solver-2.0/install_js/lib/cmake/Ceres)
set(OpenCV_DIR ~/shared/opencv-3.4.16/build_js/install/share/OpenCV)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)


##########################
# set flags
##########################
set(COMPILE_FLAG_TEXT "-O3 -fexceptions")
set(LINK_FLAG_TEXT "-s USE_ZLIB=1 -s DYNAMIC_EXECUTION=0 \
  -s MODULARIZE=1 -s EXPORT_NAME=createModule -s SINGLE_FILE=1 \
  -s INITIAL_MEMORY=256MB -s ALLOW_MEMORY_GROWTH=1 -s ALLOW_TABLE_GROWTH=1 \
  -s NO_DISABLE_EXCEPTION_CATCHING=1 \
  -s ERROR_ON_UNDEFINED_SYMBOLS=0 \
  -O3 -fexceptions --no-entry --no-check-features --bind"
)

## enable debug on web
if(NOT JS_DEBUG)
  set(COMPILE_FLAG_TEXT "${COMPILE_FLAG_TEXT} -g0")
  set(LINK_FLAG_TEXT "${LINK_FLAG_TEXT} -g0 -s ASSERTIONS=0")
else()
  set(LINK_FLAG_TEXT "${LINK_FLAG_TEXT} -s ASSERTIONS=1")
endif()

## enable multi-thread
if(MULTI_THREAD)
  set(COMPILE_FLAG_TEXT "${COMPILE_FLAG_TEXT} -mbulk-memory -s USE_PTHREADS=1")
  set(LINK_FLAG_TEXT "${LINK_FLAG_TEXT} -s ENVIRONMENT=web,worker,shell -s USE_PTHREADS=1 -s SHARED_MEMORY=1 -mbulk-memory")
else()
  set(LINK_FLAG_TEXT "${LINK_FLAG_TEXT} -s ENVIRONMENT=web,shell")
endif()


##########################
# build wasm module
##########################
add_executable(${PROJECT_NAME}
  src/camera_calibration.cpp
  src/ceres_parametrization.cpp
  src/estimator.cpp
  src/feature_extractor.cpp
  src/feature_tracker.cpp
  src/frame.cpp
  src/map_manager.cpp
  src/map_point.cpp
  src/mapper.cpp
  src/multi_view_geometry.cpp
  src/optimizer.cpp
  src/ov2slam.cpp
  src/slam_params.cpp
  src/visual_front_end.cpp
)

### include 
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/ceres_parametrization
)

### linking
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
    Sophus::Sophus
  PRIVATE
    ${OpenCV_LIBS}
    Ceres::ceres
)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${COMPILE_FLAG_TEXT})

set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS ${LINK_FLAG_TEXT})


##########################
# adapt to js env
##########################
add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND echo "export default createModule;" >> ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.js
  VERBATIM
)

