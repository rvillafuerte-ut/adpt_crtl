#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// --- VARIABLES DE ESTADO ---
// Definimos tamaños: 2 Articulaciones (Joints), 5 Parámetros
const int N_JOINTS = 2;
const int N_PARAMS = 5;

// --- 1. CÁLCULO DE TRAYECTORIAS DE REFERENCIA (Slotine) ---
// Esta función calcula las señales "virtuales" que usa el regresor
// para no ensuciar el control con ruido de aceleración real.
struct RefSignals {
    Vector2d dqr;  // Velocidad de Referencia
    Vector2d ddqr; // Aceleración de Referencia
    Vector2d s;    // Variable Deslizante (Error compuesto)
};

RefSignals calculate_references(Vector2d q, Vector2d dq, 
                                Vector2d q_des, Vector2d dq_des, Vector2d ddq_des, 
                                Matrix2d Lambda) {
    RefSignals refs;
    
    // Error de seguimiento (q - q_deseado)
    Vector2d q_tilde = q - q_des;
    Vector2d dq_tilde = dq - dq_des;

    // 1. Velocidad de Referencia: dqr = dq_des - Lambda * q_tilde
    refs.dqr = dq_des - Lambda * q_tilde;

    // 2. Aceleración de Referencia: ddqr = ddq_des - Lambda * dq_tilde
    // NOTA: Aquí usamos dq_tilde (velocidad), NO aceleración real. ¡Esto evita el ruido!
    refs.ddqr = ddq_des - Lambda * dq_tilde;

    // 3. Variable Deslizante: s = dq - dqr
    refs.s = dq - refs.dqr;

    return refs;
}

// --- 2. CONSTRUCCIÓN DE LA MATRIZ REGRESORA Y ---
// Esta es la parte crítica. Mapea la dinámica no lineal a la forma lineal Y*theta.
MatrixXd build_regressor(Vector2d q, Vector2d dq, Vector2d dqr, Vector2d ddqr) {
    MatrixXd Y(N_JOINTS, N_PARAMS);
    
    // Pre-cálculo de senos y cosenos para eficiencia y claridad
    double c2 = cos(q(1));
    double s2 = sin(q(1));
    double s1 = sin(q(0));
    double s12 = sin(q(0) + q(1));

    // Desempaquetar variables para que las fórmulas sean legibles
    double dq1 = dq(0);
    double dq2 = dq(1);
    double dqr1 = dqr(0);
    double dqr2 = dqr(1);
    double ddqr1 = ddqr(0);
    double ddqr2 = ddqr(1);

    // --- DEFINICIÓN DE PARÁMETROS THETA (Para referencia mental) ---
    // theta1 = m1*l1^2 + m2*l1^2 + m2*l2^2  (Inercia base)
    // theta2 = m2*l1*l2                     (Acople inercial)
    // theta3 = m2*l2^2                      (Inercia eslabón 2)
    // theta4 = (m1+m2)*g*l1                 (Gravedad 1)
    // theta5 = m2*g*l2                      (Gravedad 2)

    // --- FILA 1 (Torque Articulación 1) ---
    
    // Col 1: Coeficiente de Theta1 (Inercia base pura)
    // Multiplica a la aceleración referencia eje 1
    Y(0, 0) = ddqr1; 

    // Col 2: Coeficiente de Theta2 (Términos de Acople y Coriolis)
    // OJO AQUI: Este es el término más propenso a errores.
    // Fórmula: 2*c2*ddqr1 + c2*ddqr2 - s2*dq2*dqr1 - s2*(dq1 + dq2)*dqr2
    Y(0, 1) = (2.0 * c2 * ddqr1) + (c2 * ddqr2) - (s2 * dq2 * dqr1) - (s2 * (dq1 + dq2) * dqr2);

    // Col 3: Coeficiente de Theta3 (Inercia eslabón 2 reflejada)
    Y(0, 2) = ddqr2;

    // Col 4: Coeficiente de Theta4 (Gravedad 1)
    Y(0, 3) = s1; // g ya está incluido en el parámetro theta

    // Col 5: Coeficiente de Theta5 (Gravedad 2)
    Y(0, 4) = s12;

    // --- FILA 2 (Torque Articulación 2) ---

    // Col 1: Theta1 no afecta al eje 2
    Y(1, 0) = 0.0;

    // Col 2: Coeficiente de Theta2
    // Fórmula: c2*ddqr1 + s2*dq1*dqr1
    Y(1, 1) = (c2 * ddqr1) + (s2 * dq1 * dqr1);

    // Col 3: Coeficiente de Theta3
    // Afecta a la suma de aceleraciones
    Y(1, 2) = ddqr1 + ddqr2;

    // Col 4: Theta4 (Gravedad 1) no afecta al eje 2
    Y(1, 3) = 0.0;

    // Col 5: Theta5 (Gravedad 2)
    Y(1, 4) = s12;

    return Y;
}

// --- MAIN LOOP DE EJEMPLO ---
int main() {
    Matrix2d Lambda = Matrix2d::Identity() * 5.0;
    Matrix2d KD = Matrix2d::Identity() * 15.0;
    VectorXd theta_hat(N_PARAMS); theta_hat << 0.2, 0.2, 0.2, 0.2, 0.2;
    VectorXd theta_real(N_PARAMS); // Parámetros reales
    // Parámetros físicos (ejemplo)
    double m1 = 1.5, m2 = 1.0, l1 = 0.5, l2 = 0.4, g = 9.81;
    theta_real(0) = m1*l1*l1 + m2*l1*l1 + m2*l2*l2;
    theta_real(1) = m2*l1*l2;
    theta_real(2) = m2*l2*l2;
    theta_real(3) = (m1+m2)*g*l1;
    theta_real(4) = m2*g*l2;

    Vector2d q(0.4, 0.25);
    Vector2d dq = Vector2d::Zero();
    Vector2d ddq = Vector2d::Zero();
    Vector2d tau = Vector2d::Zero();

    double t = 0.0, dt = 0.001;
    MatrixXd Gamma = MatrixXd::Identity(N_PARAMS, N_PARAMS) * 5.0; // Adaptación

    cout << "t,q1,q2,ref1,ref2,tau1,tau2,th1,th2,th3,th4,th5,th1_true,th2_true,th3_true,th4_true,th5_true" << endl;

    while (t < 150.0) {
        // Trayectoria "Rica" (Suma de senos)
        // Frecuencias: 0.5, 1.3, 2.1 Hz (mezcladas)
        double w1 = 0.5, w2 = 1.3, w3 = 2.1;
        
        // EJE 1: Movimiento complejo
        double q1_d = 0.6 * sin(w1*t) + 0.15 * cos(w3*t);
        double dq1_d = 0.6*w1*cos(w1*t) - 0.15*w3*sin(w3*t);
        double ddq1_d = -0.6*w1*w1*sin(w1*t) - 0.15*w3*w3*cos(w3*t);

        // EJE 2: Movimiento complejo desfasado
        double q2_d = 0.5 * cos(w2*t) + 0.1 * sin(w1*t);
        double dq2_d = -0.5*w2*sin(w2*t) + 0.1*w1*cos(w1*t);
        double ddq2_d = -0.5*w2*w2*cos(w2*t) - 0.1*w1*w1*sin(w1*t);

        Vector2d q_des, dq_des, ddq_des;
        q_des << q1_d, q2_d;
        dq_des << dq1_d, dq2_d;
        ddq_des << ddq1_d, ddq2_d;

        

        RefSignals refs = calculate_references(q, dq, q_des, dq_des, ddq_des, Lambda);
        MatrixXd Y = build_regressor(q, dq, refs.dqr, refs.ddqr);

        tau = Y * theta_hat - KD * refs.s;
        tau = tau.cwiseMax(-50.0).cwiseMin(50.0);

        // Dinámica real M(q) ddq + C(q,dq)dq + G(q) = tau
        double q1 = q(0), q2 = q(1);
        double dq1 = dq(0), dq2 = dq(1);
        double c2 = cos(q2), s2 = sin(q2);
        // Matriz de inercia
        double M11 = theta_real(0) + 2.0*theta_real(1)*c2; // expandido
        double M12 = theta_real(2) + theta_real(1)*c2;
        double M21 = M12;
        double M22 = theta_real(2);
        Matrix2d M; M << M11, M12, M21, M22;
        // Coriolis+centrífugo
        double h = -theta_real(1)*s2; // = -m2*l1*l2*sin(q2)
        double C1 = h*dq2*(2.0*dq1 + dq2);
        double C2 = -h*dq1*dq1;
        Vector2d C; C << C1, C2;
        // Gravedad
        double G1 = theta_real(3)*sin(q1) + theta_real(4)*sin(q1+q2);
        double G2 = theta_real(4)*sin(q1+q2);
        Vector2d Gvec; Gvec << G1, G2;

        ddq = M.inverse() * (tau - C - Gvec);
        dq += ddq * dt;
        q += dq * dt;

        // Ley de adaptación (Slotine-Li): dot(theta)= -Gamma Y^T s
        theta_hat += -Gamma * Y.transpose() * refs.s * dt;

        if (int(t*1000) % 10 == 0) {
              cout << t << "," << q(0) << "," << q(1) << "," << q_des(0) << "," << q_des(1)
                  << "," << tau(0) << "," << tau(1) << "," << theta_hat(0) << "," << theta_hat(1)
                  << "," << theta_hat(2) << "," << theta_hat(3) << "," << theta_hat(4)
                  << "," << theta_real(0) << "," << theta_real(1) << "," << theta_real(2)
                  << "," << theta_real(3) << "," << theta_real(4) << endl;
        }

        t += dt;
    }
    return 0;
}