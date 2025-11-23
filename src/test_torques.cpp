#include <iostream>
#include <Eigen/Dense>
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

using namespace Eigen;

int main() {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("open_x.urdf", model);
    pinocchio::Data data(model);
    
    // Test what torques are needed for a simple trajectory
    VectorXd q = VectorXd::Zero(model.nq);
    VectorXd dq = VectorXd::Zero(model.nv);
    VectorXd ddq = VectorXd::Zero(model.nv);
    
    std::cout << "=== TORQUE REQUIREMENTS TEST ===\n\n";
    
    // Test 1: Gravity compensation at q=0
    pinocchio::rnea(model, data, q, dq, ddq);
    std::cout << "Gravity torques at q=0:\n" << data.tau.transpose() << "\n\n";
    
    // Test 2: At q = [0.4, 0.4, 0.4, 0.4] (realistic angles)
    q = VectorXd::Constant(model.nq, 0.4);
    pinocchio::rnea(model, data, q, dq, ddq);
    std::cout << "Gravity torques at q=0.4rad:\n" << data.tau.transpose() << "\n\n";
    
    // Test 3: With velocity
    dq = VectorXd::Constant(model.nv, 1.0);
    pinocchio::rnea(model, data, q, dq, ddq);
    std::cout << "Torques with dq=1rad/s:\n" << data.tau.transpose() << "\n\n";
    
    // Test 4: With acceleration
    ddq = VectorXd::Constant(model.nv, 2.0);
    pinocchio::rnea(model, data, q, dq, ddq);
    std::cout << "Torques with ddq=2rad/s²:\n" << data.tau.transpose() << "\n\n";
    
    std::cout << "Max torque needed: " << data.tau.cwiseAbs().maxCoeff() << " Nm\n";
    
    return 0;
}
