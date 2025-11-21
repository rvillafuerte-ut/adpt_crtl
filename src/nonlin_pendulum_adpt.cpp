#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

const double dt = 0.001;
const double T_FINAL = 35.0;
const double LAMBDA = 8.0;
const double FORGETTING = 0.995;

Vector2d x_real(0.3, 0.0);  // [theta, theta_dot]
Vector2d x_filt(0.0, 0.0);
double u_filt = 0.0;
double sin_filt = 0.0;

void simulate_pendulum(double u, double m, double L, double b) {
    const double g = 9.81;
    double theta = x_real(0);
    double theta_dot = x_real(1);
    
    // theta_ddot = -(g/L)*sin(theta) - (b/mL^2)*theta_dot + (1/mL^2)*u
    double I = m * L * L;
    double theta_ddot = -(g/L) * sin(theta) - (b/I) * theta_dot + (1.0/I) * u;
    
    x_real(1) += theta_ddot * dt;
    x_real(0) += x_real(1) * dt;
}

int main() {
    Vector4d theta_hat(0.0, -1.0, -5.0, 1.0);  // [a21_lin, a22, a21_nl, b2]
    Matrix4d P = Matrix4d::Identity() * 1000.0;

    double wn = 5.0, zeta = 0.8;
    double alpha1_des = 2.0 * zeta * wn;
    double alpha0_des = wn * wn;

    double t = 0.0, u = 0.0, ref = 0.0;
    double m = 1.0, L = 1.0, b = 0.1;

    cout << "t,ref,theta,u,est_a21_lin,est_a21_nl,est_b2,K1,K2" << endl;

    while (t < T_FINAL) {
        if (t > 3.0 && m == 1.0) m = 2.5;  // Cambio de inercia
        
        if (t > 3.0) ref = 0.5;  // Step reference
        if (t > 8.0) ref = -0.5; 
        // Filtros SVF
        x_filt += LAMBDA * (x_real - x_filt) * dt;
        u_filt += LAMBDA * (u - u_filt) * dt;
        sin_filt += LAMBDA * (sin(x_real(0)) - sin_filt) * dt;

        // RLS: theta_ddot = a21_lin*theta + a22*theta_dot + a21_nl*sin(theta) + b2*u
        Vector2d dx_filt = LAMBDA * (x_real - x_filt);
        Vector4d phi; 
        phi << x_filt(0), x_filt(1), sin_filt, u_filt;
        double error = dx_filt(1) - phi.dot(theta_hat);

        if (abs(error) > 0.001) {
            Vector4d K_rls = (P * phi) / (FORGETTING + phi.dot(P * phi));
            theta_hat += K_rls * error;
            P = (Matrix4d::Identity() - K_rls * phi.transpose()) * P / FORGETTING;
        }

        // Linealización por retroalimentación: u = v - (a21_nl/b2)*sin(theta)
        double hat_b2 = (abs(theta_hat(3)) < 0.01) ? 0.01 : theta_hat(3);
        double u_nl = -(theta_hat(2) / hat_b2) * sin(x_real(0));
        
        // Sistema linealizado: theta_ddot ≈ a21_lin*theta + a22*theta_dot + b2*v
        double k1 = (alpha0_des + theta_hat(0)) / hat_b2;
        double k2 = (alpha1_des + theta_hat(1)) / hat_b2;
        double Kr = alpha0_des / hat_b2;

        double v = -k1 * x_real(0) - k2 * x_real(1) + Kr * ref;
        u = v + u_nl;
        u = max(-30.0, min(30.0, u));

        simulate_pendulum(u, m, L, b);
        t += dt;

        if (int(t*1000) % 50 == 0) {
            cout << t << "," << ref << "," << x_real(0) << "," << u << "," 
                 << theta_hat(0) << "," << theta_hat(2) << "," << theta_hat(3) << "," 
                 << k1 << "," << k2 << endl;
        }
    }

    return 0;
}
