CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(calibcamodo)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)

# c++ 11 support for g2o, multi-thread and shared_ptr
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")

# Find third party packages
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(Cholmod REQUIRED)
FIND_PACKAGE(G2O REQUIRED)
# glog
find_package( Glog REQUIRED )
include_directories( ${GLOG_INCLUDE_DIRS} )
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
FIND_PACKAGE(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    roscpp
    rospy
    tf
    roslib
    cv_bridge
    cmake_modules
    image_transport
    slam_car
)

CATKIN_PACKAGE()

INCLUDE_DIRECTORIES(
    ./src/
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${catkin_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${Cholmod_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
)

LIST(APPEND LINK_LIBS
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${GLOG_LIBRARY}
)

LIST(APPEND G2O_LIBS
    cxsparse
    cholmod
    g2o_cli 
    g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
    g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
    g2o_types_sba g2o_types_slam3d g2o_csparse_extension
    g2o_solver_dense g2o_stuff
    g2o_types_sclam2d g2o_parser g2o_solver_pcg
    g2o_types_data g2o_types_sim3
)

## Declare a C++ executable
AUX_SOURCE_DIRECTORY(./src SRC_DIR )
AUX_SOURCE_DIRECTORY(./src/aruco SRC_DIR_ARUCO)
AUX_SOURCE_DIRECTORY(./src/g2o SRC_DIR_G2O)
AUX_SOURCE_DIRECTORY(./src/ros SRC_DIR_ROS)
AUX_SOURCE_DIRECTORY(./src/orb SRC_DIR_ORB)
AUX_SOURCE_DIRECTORY(./src/core SRC_DIR_CORE)

AUX_SOURCE_DIRECTORY(./src/slide_opt SRC_DIR_SLIDE_OPT)

## 添加库: serial_api
add_library(slide_opt_shared SHARED
    ${SRC_DIR_SLIDE_OPT}
)

ADD_LIBRARY(calibcamodo SHARED
    ${SRC_DIR_ARUCO}
    ${SRC_DIR_G2O}
    ${SRC_DIR_ROS}
    ${SRC_DIR_ORB}
    ${SRC_DIR_CORE}
)
TARGET_LINK_LIBRARIES(calibcamodo
    ${LINK_LIBS}
    ${G2O_LIBS}
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
)

ADD_EXECUTABLE(
    calib_aruco
    ./src/calib_aruco.cpp
)
TARGET_LINK_LIBRARIES(calib_aruco
    calibcamodo
    ${LINK_LIBS}
)

ADD_EXECUTABLE(
    calib_orb
    ./src/calib_orb.cpp
)
TARGET_LINK_LIBRARIES(calib_orb
    calibcamodo
    ${LINK_LIBS}
)

ADD_EXECUTABLE(
    calib_orb_filter
    ./src/calib_orb_filter.cpp
)
TARGET_LINK_LIBRARIES(calib_orb_filter
    calibcamodo
    ${LINK_LIBS}
)

ADD_EXECUTABLE(
    detect_aruco
    ./src/detect_aruco.cpp
)
TARGET_LINK_LIBRARIES(detect_aruco
    calibcamodo
    ${LINK_LIBS}
)

ADD_EXECUTABLE(
    calib_aruco_online
    ./src/calib_aruco_online.cpp
)
TARGET_LINK_LIBRARIES(calib_aruco_online
    calibcamodo
    ${LINK_LIBS}
    slide_opt_shared
)
