#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

const double dt = 0.001;
const double T_FINAL = 100.0;
const double LAMBDA = 10.0;
const double FORGETTING = 0.99;

Vector4d x_real = Vector4d::Zero();
Vector4d x_filt = Vector4d::Zero();
Vector2d u_filt = Vector2d::Zero();

void simulate_plant(const Vector2d& u, double m1, double m2, double k1, double k2, double c1, double c2) {
    // Sistema MIMO: 2 masas acopladas con 2 actuadores
    // x = [p1, v1, p2, v2]'
    // dx = Ax + Bu
    Matrix4d A;
    A << 0, 1, 0, 0,
         -(k1+k2)/m1, -c1/m1, k2/m1, 0,
         0, 0, 0, 1,
         k2/m2, 0, -k2/m2, -c2/m2;
    
    Matrix<double, 4, 2> B;
    B << 0, 0,
         1/m1, 0,
         0, 0,
         0, 1/m2;
    
    x_real += (A * x_real + B * u) * dt;
}

int main() {
    // Estimador 1: fila 2 de A y B -> [a21, a22, a23, a24, b21, b22]
    VectorXd theta1(6);
    theta1 << -3.0, -1.0, 1.0, 0.0, 1.0, 0.0;
    MatrixXd P1 = MatrixXd::Identity(6, 6) * 1000.0;
    
    // Estimador 2: fila 4 de A y B -> [a41, a42, a43, a44, b41, b42]
    VectorXd theta2(6);
    theta2 << 1.0, 0.0, -1.0, -1.0, 0.0, 1.0;
    MatrixXd P2 = MatrixXd::Identity(6, 6) * 1000.0;

    // LQR: Q y R
    Matrix4d Q = Matrix4d::Identity();
    Q(0,0) = 10000.0; Q(2,2) = 10000.0;  // Penalizar posiciones
    Q(1,1) = 1.0;  Q(3,3) = 1.0;   // Penalizar velocidades
    
    Matrix2d R = Matrix2d::Identity() * 0.01;

    double t = 0.0;
    Vector2d u = Vector2d::Zero();
    Vector2d ref; ref << 1.0, 0.5;
    Vector2d ramp; ramp << 1.0, 0.5;
    
    double m1 = 1.0, m2 = 1.0;
    double k1 = 2.0, k2 = 1.0;
    double c1 = 0.5, c2 = 0.3;

    cout << "t,ref1,ref2,p1,p2,u1,u2,est_a21,est_b21,est_a43,est_b42" << endl;

    while (t < T_FINAL) {
        if (t > 6.0 && m1 == 1.0) { m1 = 2.0; m2 = 1.5; }
        ref = ramp*sin(0.2*t);  // Referencias sinusoidales

        // Filtros SVF
        x_filt += LAMBDA * (x_real - x_filt) * dt;
        u_filt += LAMBDA * (u - u_filt) * dt;

        // RLS Estimador 1: dx2/dt = a21*x1 + a22*x2 + a23*x3 + a24*x4 + b21*u1 + b22*u2
        Vector4d dx_filt = LAMBDA * (x_real - x_filt);
        VectorXd phi1(6);
        phi1 << x_filt(0), x_filt(1), x_filt(2), x_filt(3), u_filt(0), u_filt(1);
        double error1 = dx_filt(1) - phi1.dot(theta1);

        if (abs(error1) > 0.0001) {
            VectorXd K_rls1 = (P1 * phi1) / (FORGETTING + phi1.dot(P1 * phi1));
            theta1 += K_rls1 * error1;
            P1 = (MatrixXd::Identity(6,6) - K_rls1 * phi1.transpose()) * P1 / FORGETTING;
        }

        // RLS Estimador 2: dx4/dt = a41*x1 + a42*x2 + a43*x3 + a44*x4 + b41*u1 + b42*u2
        VectorXd phi2(6);
        phi2 << x_filt(0), x_filt(1), x_filt(2), x_filt(3), u_filt(0), u_filt(1);
        double error2 = dx_filt(3) - phi2.dot(theta2);

        if (abs(error2) > 0.0001) {
            VectorXd K_rls2 = (P2 * phi2) / (FORGETTING + phi2.dot(P2 * phi2));
            theta2 += K_rls2 * error2;
            P2 = (MatrixXd::Identity(6,6) - K_rls2 * phi2.transpose()) * P2 / FORGETTING;
        }

        // Reconstruir A y B estimados
        Matrix4d A_hat = Matrix4d::Zero();
        A_hat.row(0) << 0, 1, 0, 0;
        A_hat.row(1) << theta1(0), theta1(1), theta1(2), theta1(3);
        A_hat.row(2) << 0, 0, 0, 1;
        A_hat.row(3) << theta2(0), theta2(1), theta2(2), theta2(3);

        Matrix<double, 4, 2> B_hat;
        B_hat << 0, 0,
                 theta1(4), theta1(5),
                 0, 0,
                 theta2(4), theta2(5);

        // --- CORRECCIÓN: DISCRETIZAR PARA LQR ---
        // El solver iterativo de abajo es para tiempo discreto (DARE),
        // pero A_hat y B_hat son continuas. Debemos convertirlas.
        Matrix4d I = Matrix4d::Identity();
        Matrix4d A_d = I + A_hat * dt;            // A_discreta approx
        Matrix<double, 4, 2> B_d = B_hat * dt;    // B_discreta approx

        // Resolver Riccati Discreto (DARE) iterativo
        // Usamos A_d y B_d en lugar de A_hat y B_hat
        Matrix4d P_lqr = Q; // Inicializar con Q para convergencia rápida
        
        // Nota: 50 iteraciones es mucho para un microcontrolador, 
        // en PC está bien. En real-time se suele hacer 1 iteración por ciclo.
        for (int i = 0; i < 20; i++) { 
            Matrix2d R_total = R + B_d.transpose() * P_lqr * B_d;
            Matrix<double, 2, 4> K_temp = R_total.inverse() * B_d.transpose() * P_lqr * A_d;
            P_lqr = Q + A_d.transpose() * P_lqr * (A_d - B_d * K_temp);
        }
        
        // Calcular Ganancia K Final (con matrices discretas)
        Matrix<double, 2, 4> K_lqr = (R + B_d.transpose() * P_lqr * B_d).inverse() 
                                     * B_d.transpose() * P_lqr * A_d;

        // Control con referencia
        // u = -K * (x - ref)
        Vector4d error_state = x_real;
        error_state(0) -= ref(0); // Error de posición masa 1
        error_state(2) -= ref(1); // Error de posición masa 2
        
        u = -K_lqr * error_state;
        u = u.cwiseMax(-50.0).cwiseMin(50.0); // Aumenté un poco la saturación

        simulate_plant(u, m1, m2, k1, k2, c1, c2);
        t += dt;

        if (int(t*1000) % 50 == 0) {
            cout << t << "," << ref(0) << "," << ref(1) << "," 
                 << x_real(0) << "," << x_real(2) << "," 
                 << u(0) << "," << u(1) << "," 
                 << theta1(0) << "," << theta1(4) << "," 
                 << theta2(2) << "," << theta2(5) << endl;
        }
    }

    return 0;
}
