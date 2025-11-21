#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

const double dt = 0.001;
const double T_FINAL = 10.0;
const double LAMBDA = 10.0;
const double FORGETTING = 0.99;

Vector2d x_real(0.0, 0.0);
Vector2d x_filt(0.0, 0.0);
double u_filt = 0.0;

void simulate_plant(double u, double m, double c, double k) {
    Matrix2d A; A << 0, 1, -k/m, -c/m;
    Vector2d B; B << 0, 1/m;
    x_real += (A * x_real + B * u) * dt;
}

int main() {
    Vector3d theta_hat(-1.0, -1.0, 2.0);  // [a21, a22, b2]
    Matrix3d P = Matrix3d::Identity() * 1000.0;

    double wn = 4.0, zeta = 0.707;
    double alpha1_des = 2.0 * zeta * wn;
    double alpha0_des = wn * wn;

    double t = 0.0, u = 0.0, ref = 1.0;
    double m = 1.0, c = 0.5, k = 2.0;

    cout << "t,ref,pos,u,est_a21,est_b2,K1,K2" << endl;

    while (t < T_FINAL) {
        if (t > 5.0 && m == 1.0) m = 3.0;  // Cambio de dinámica
        if (t > 5.0) ref = -3.0;
        // Filtros SVF
        x_filt += LAMBDA * (x_real - x_filt) * dt;
        u_filt += LAMBDA * (u - u_filt) * dt;

        // RLS: estimar theta = [a21, a22, b2]
        Vector2d dx_filt = LAMBDA * (x_real - x_filt);
        Vector3d phi; phi << x_filt(0), x_filt(1), u_filt;
        double error = dx_filt(1) - phi.dot(theta_hat);

        if (abs(error) > 0.0001) {
            Vector3d K_rls = (P * phi) / (FORGETTING + phi.dot(P * phi));
            theta_hat += K_rls * error;
            P = (Matrix3d::Identity() - K_rls * phi.transpose()) * P / FORGETTING;
        }

        // Asignación de polos
        double hat_b2 = (abs(theta_hat(2)) < 0.01) ? 0.01 : theta_hat(2);
        double k1 = (alpha0_des + theta_hat(0)) / hat_b2;
        double k2 = (alpha1_des + theta_hat(1)) / hat_b2;
        double Kr = alpha0_des / hat_b2;

        u = -k1 * x_real(0) - k2 * x_real(1) + Kr * ref;
        u = max(-20.0, min(20.0, u));

        simulate_plant(u, m, c, k);
        t += dt;

        if (int(t*1000) % 50 == 0) {
            cout << t << "," << ref << "," << x_real(0) << "," << u << "," 
                 << theta_hat(0) << "," << theta_hat(2) << "," 
                 << k1 << "," << k2 << endl;
        }
    }

    return 0;
}