#include <iostream>
#include <Eigen/Dense>
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/regressor.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

using namespace Eigen;
using namespace std;

int main() {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("open_x.urdf", model);
    pinocchio::Data data(model);
    
    cout << "=== MODEL INFO ===" << endl;
    cout << "nq (positions): " << model.nq << endl;
    cout << "nv (velocities): " << model.nv << endl;
    cout << "njoints: " << model.njoints << endl;
    
    // Test regressor at a non-zero configuration
    VectorXd q = VectorXd::Zero(model.nq);
    VectorXd v = VectorXd::Ones(model.nv) * 0.1;
    VectorXd a = VectorXd::Ones(model.nv) * 0.1;
    
    pinocchio::computeJointTorqueRegressor(model, data, q, v, a);
    
    cout << "\n=== REGRESSOR INFO ===" << endl;
    cout << "Regressor size: " << data.jointTorqueRegressor.rows() 
         << " x " << data.jointTorqueRegressor.cols() << endl;
    
    cout << "\nFirst row (joint 1 parameters):\n" << data.jointTorqueRegressor.row(0) << endl;
    cout << "\nColumn norms (parameter importance):" << endl;
    for(int i=0; i<data.jointTorqueRegressor.cols(); i++) {
        cout << "Param " << i << ": " << data.jointTorqueRegressor.col(i).norm() << endl;
    }
    
    cout << "\n=== JOINT NAMES ===" << endl;
    for(size_t i=0; i<model.names.size(); i++) {
        cout << "Joint " << i << ": " << model.names[i] << endl;
    }
    
    return 0;
}
