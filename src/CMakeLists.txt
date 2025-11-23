cmake_minimum_required(VERSION 3.10)
project(adpt_crtl)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

add_executable(lin_ss_adpt lin_ss_adpt.cpp)
add_executable(nonlin_pendulum_adpt nonlin_pendulum_adpt.cpp)
add_executable(mimo_lqr_adpt mimo_lqr_adpt.cpp)
add_executable(2dof_robot_adpt 2dof_robot_adpt.cpp)
add_executable(4dof_robot_adpt 4dof_robot_adpt.cpp)
add_executable(4dof_robot_adpt_manual 4dof_robot_adpt_manual.cpp)

target_link_libraries(lin_ss_adpt PRIVATE Eigen3::Eigen)
target_link_libraries(nonlin_pendulum_adpt PRIVATE Eigen3::Eigen)
target_link_libraries(mimo_lqr_adpt PRIVATE Eigen3::Eigen)
target_link_libraries(2dof_robot_adpt PRIVATE Eigen3::Eigen)
target_link_libraries(4dof_robot_adpt PRIVATE Eigen3::Eigen pinocchio::pinocchio)
target_link_libraries(4dof_robot_adpt_manual PRIVATE Eigen3::Eigen pinocchio::pinocchio)

# Copiar URDF al directorio de build para ejecución directa
set(URDF_FILE ${CMAKE_SOURCE_DIR}/../urdf/open_x.urdf)
if(EXISTS ${URDF_FILE})
    configure_file(${URDF_FILE} ${CMAKE_CURRENT_BINARY_DIR}/open_x.urdf COPYONLY)
else()
    message(WARNING "No se encontró ${URDF_FILE}; 4dof_robot_adpt no podrá cargar el URDF.")
endif()