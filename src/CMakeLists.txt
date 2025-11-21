cmake_minimum_required(VERSION 3.10)
project(adpt_crtl)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN_DIR "$ENV{HOME}/libs_utils/eigen")


include_directories(
    ${EIGEN_DIR}
    )

add_executable(lin_ss_adpt lin_ss_adpt.cpp)
add_executable(nonlin_pendulum_adpt nonlin_pendulum_adpt.cpp)
add_executable(mimo_lqr_adpt mimo_lqr_adpt.cpp)
add_executable(2dof_robot_adpt 2dof_robot_adpt.cpp)