#include <iostream>
#include <Eigen/Dense>
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/regressor.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

using namespace Eigen;

int main() {
    pinocchio::Model model;
    pinocchio::urdf::buildModel("open_x.urdf", model);
    pinocchio::Data data(model);
    
    std::cout << "=== REGRESSOR VERIFICATION ===\n\n";
    
    // Random configuration
    VectorXd q = VectorXd::Random(model.nq) * 0.5;
    VectorXd dq = VectorXd::Random(model.nv) * 0.5;
    VectorXd ddq = VectorXd::Random(model.nv) * 0.5;
    
    // Compute true torques with RNEA
    pinocchio::rnea(model, data, q, dq, ddq);
    VectorXd tau_true = data.tau;
    
    // Compute regressor
    pinocchio::computeJointTorqueRegressor(model, data, q, dq, ddq);
    
    // Extract TRUE parameters from model
    VectorXd theta_real(data.jointTorqueRegressor.cols());
    theta_real.setZero();
    
    // IMPORTANTE: Pinocchio's regressor expects parameters in a specific order
    // For each body: [m, m*c_x, m*c_y, m*c_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz]
    // where c is the center of mass
    
    int param_idx = 0;
    for(size_t i=1; i<model.inertias.size(); i++) {  // Skip universe
        const auto& I = model.inertias[i];
        double m = I.mass();
        Vector3d mc = I.lever();  // m * com
        Matrix3d Ic = I.inertia().matrix();  // Inertia at COM
        
        theta_real(param_idx + 0) = m;
        theta_real(param_idx + 1) = mc(0);
        theta_real(param_idx + 2) = mc(1);
        theta_real(param_idx + 3) = mc(2);
        theta_real(param_idx + 4) = Ic(0,0);
        theta_real(param_idx + 5) = Ic(0,1);
        theta_real(param_idx + 6) = Ic(0,2);
        theta_real(param_idx + 7) = Ic(1,1);
        theta_real(param_idx + 8) = Ic(1,2);
        theta_real(param_idx + 9) = Ic(2,2);
        
        std::cout << "Body " << i << " (" << model.names[i] << "):\n";
        std::cout << "  m = " << m << "\n";
        std::cout << "  mc = " << mc.transpose() << "\n";
        
        param_idx += 10;
    }
    
    // Reconstruct torques from regressor
    VectorXd tau_reconstructed = data.jointTorqueRegressor * theta_real;
    
    std::cout << "\n=== VERIFICATION ===\n";
    std::cout << "True torques (RNEA):\n" << tau_true.transpose() << "\n\n";
    std::cout << "Reconstructed (Y*theta):\n" << tau_reconstructed.transpose() << "\n\n";
    std::cout << "Error norm: " << (tau_true - tau_reconstructed).norm() << "\n";
    
    if ((tau_true - tau_reconstructed).norm() < 1e-10) {
        std::cout << "\n✓ REGRESSOR IS CORRECT!\n";
    } else {
        std::cout << "\n✗ REGRESSOR MISMATCH - Parameter structure issue\n";
    }
    
    return 0;
}
