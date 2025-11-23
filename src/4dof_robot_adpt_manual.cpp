#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/regressor.hpp" 

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// --- CONFIGURACIÓN ---
const double DT = 0.001;
const double T_END = 150.0;

struct RefSignals { VectorXd dqr; VectorXd ddqr; VectorXd s; };

RefSignals calculate_references(const VectorXd& q, const VectorXd& dq, 
                                const VectorXd& q_des, const VectorXd& dq_des, const VectorXd& ddq_des, 
                                const MatrixXd& Lambda) {
    RefSignals refs;
    VectorXd q_tilde = q - q_des;
    VectorXd dq_tilde = dq - dq_des;
    refs.dqr = dq_des - Lambda * q_tilde;
    refs.ddqr = ddq_des - Lambda * dq_tilde;
    refs.s = dq - refs.dqr;
    return refs;
}

int main(int argc, char ** argv) {
    
    // 1. CARGAR MODELO
    const string urdf_filename = "open_x.urdf";
    pinocchio::Model model;
    try {
        pinocchio::urdf::buildModel(urdf_filename, model);
    } catch (const std::exception& e) {
        cerr << "ERROR CARGANDO URDF: " << e.what() << endl;
        return -1;
    }
    pinocchio::Data data(model);

    int nq = model.nq;
    int nv = model.nv;

    // 2. VARIABLES DE ESTADO
    VectorXd q = VectorXd::Zero(nq);
    VectorXd dq = VectorXd::Zero(nv);
    VectorXd tau = VectorXd::Zero(nv);

    // 3. AUTO-DETECCIÓN DE DIMENSIONES (CRÍTICO PARA EVITAR CRASH)
    // Ejecutamos el regresor una vez con ceros para ver qué tamaño devuelve Pinocchio
    VectorXd dummy_v = VectorXd::Zero(nv);
    pinocchio::computeJointTorqueRegressor(model, data, q, dummy_v, dummy_v);
    
    // Preguntamos el tamaño real a la matriz
    int real_cols = data.jointTorqueRegressor.cols();
    int real_rows = data.jointTorqueRegressor.rows();

    cout << "--- DIAGNÓSTICO PINOCCHIO ---" << endl;
    cout << "Joints (Model): " << model.njoints << endl;
    cout << "DoF (NV): " << nv << endl;
    cout << "Regresor Generado: " << real_rows << " x " << real_cols << endl;
    
    // Crear theta_hat con el tamaño EXACTO que pide el regresor
    VectorXd theta_hat = VectorXd::Zero(real_cols);

    // 4. INICIALIZACIÓN DE PARÁMETROS
    // Inicializar todo en un valor pequeño no-cero
    theta_hat.fill(0.001);

    // Intentar poner masa en los lugares lógicos.
    // En Pinocchio estándar, los bloques son de 10 en 10.
    // El último joint (punta) suele estar al final.
    for(int k = 0; k < real_cols; k+=10) {
        if (k + 9 < real_cols) {
            theta_hat(k) = 0.1;   // Masa estimada 0.1 kg
            theta_hat(k+4) = 1e-4; // Ixx pequeño
            theta_hat(k+7) = 1e-4; // Iyy pequeñ
            theta_hat(k+9) = 1e-4; // Izz pequeño
        }
    }

    // 5. GANANCIAS
    MatrixXd Lambda = MatrixXd::Identity(nv, nv) * 0.20;
    MatrixXd KD = MatrixXd::Identity(nv, nv) * 0.10;
    
    // Gamma pequeño para estabilidad inicial
    MatrixXd Gamma = MatrixXd::Identity(real_cols, real_cols) * 0.5; 
    
    // Sigma (Leakage) para evitar deriva de parámetros no excitados
    double sigma = 0.05; 

    double t = 0.0;
    cout << "Iniciando Bucle de Control..." << endl;

    // ARCHIVO LOG (ampliado con trayectorias reales y deseadas)
    ofstream log("debug_data.csv");
    // Header: t, q[1..nv], q_des[1..nv], q_err[1..nv], tau_model[1..nv], tau_pd[1..nv], tau_total[1..nv], mass_est[1..nv]
    log << "t";
    for(int i=0;i<nv;i++) log << ",q" << (i+1);
    for(int i=0;i<nv;i++) log << ",q_des" << (i+1);
    for(int i=0;i<nv;i++) log << ",q_err" << (i+1);
    for(int i=0;i<nv;i++) log << ",tau_model" << (i+1);
    for(int i=0;i<nv;i++) log << ",tau_pd" << (i+1);
    for(int i=0;i<nv;i++) log << ",tau_total" << (i+1);
    for(int i=0;i<nv;i++) log << ",mass_est" << (i+1);
    log << "\n";

    while (t < T_END) {
        
        // A. TRAYECTORIA (Soft Start: 1-cos para v(0)=0)
        // A. GENERAR TRAYECTORIA DE EXCITACIÓN (Suma de 3 Cosinos)
        // --------------------------------------------------------------
        // Usamos la forma q = A * (1 - cos(wt)) para asegurar que q(0)=0 y dq(0)=0.
        // Sumamos 3 armónicos para garantizar "Excitación Persistente".
        
        VectorXd q_des(nq), dq_des(nv), ddq_des(nv);
        
        for(int i=0; i<nq; i++) {
            // Frecuencia base distinta para cada joint (evita que se muevan sincronizados)
            // Joint 0: 0.5Hz, Joint 1: 0.7Hz, etc.
            double w_base = 0.5 + 0.2 * i; 
            
            // --- COMPONENTE 1: Movimiento Lento y Amplio (Inercia Base) ---
            double A1 = 0.4; 
            double w1 = w_base; 
            
            // --- COMPONENTE 2: Movimiento Medio (Coriolis) ---
            double A2 = 0.15; 
            double w2 = w_base * 2.3; // 2.3 es primo relativo (evita resonancias simples)
            
            // --- COMPONENTE 3: Vibración Rápida y Corta (Inercia Local/Fricción) ---
            double A3 = 0.05; 
            double w3 = w_base * 5.1; 

            // --- CÁLCULO EXACTO DE DERIVADAS ---
            
            // 1. Posición: Suma de (1 - cos)
            q_des(i) = A1*(1.0 - cos(w1*t)) 
                     + A2*(1.0 - cos(w2*t)) 
                     + A3*(1.0 - cos(w3*t));
            
            // 2. Velocidad: Derivada de (1-cos) es sin. (Regla de la cadena: w sale fuera)
            // dq = A * w * sin(wt)
            dq_des(i) = A1*w1*sin(w1*t) 
                      + A2*w2*sin(w2*t) 
                      + A3*w3*sin(w3*t);
            
            // 3. Aceleración: Derivada de sin es cos. (Otra w sale fuera -> w^2)
            // ddq = A * w^2 * cos(wt)
            ddq_des(i) = A1*w1*w1*cos(w1*t) 
                       + A2*w2*w2*cos(w2*t) 
                       + A3*w3*w3*cos(w3*t);
        }

        // B. REFERENCIAS SLOTINE
        RefSignals refs = calculate_references(q, dq, q_des, dq_des, ddq_des, Lambda);
        
        // C. CALCULAR REGRESOR
        pinocchio::computeJointTorqueRegressor(model, data, q, refs.dqr, refs.ddqr);
        
        // D. LEY DE CONTROL
        // Ahora es IMPOSIBLE que falle la dimensión: (nv x cols) * (cols x 1)
        tau = data.jointTorqueRegressor * theta_hat - KD * refs.s;
        
        // Saturación suave
        tau = tau.cwiseMax(-8.0).cwiseMin(8.0);

        // E. SIMULACIÓN (Planta Real usando URDF)
        pinocchio::aba(model, data, q, dq, tau);
        VectorXd ddq = data.ddq;

        dq += ddq * DT;
        q = pinocchio::integrate(model, q, dq * DT);

        // F. ADAPTACIÓN ROBUSTA (Sigma Modification)
        // Ley: dot(theta) = -Gamma * (Y' * s + sigma * theta)
        VectorXd gradient = data.jointTorqueRegressor.transpose() * refs.s;
        VectorXd leakage = sigma * theta_hat; 
        
        // Anti-Windup básico: Si estamos saturados, no adaptar tanto
        bool saturated = (tau.array().abs() > 7.9).any();
        if(!saturated) {
            theta_hat += -Gamma * (gradient + leakage) * DT;
        }

        // Proyección simple
        for(int k=0; k<real_cols; k+=10) {
            if (k < real_cols) {
                if(theta_hat(k) < 0.001) theta_hat(k) = 0.001; // Masa positiva
            }
        }

        t += DT;

        // GUARDAR DATOS AL CSV (Cada 10ms para no saturar el archivo)
        if (int(t*1000) % 10 == 0) {
            VectorXd q_err = q - q_des;
            VectorXd tau_model = data.jointTorqueRegressor * theta_hat;
            VectorXd tau_pd = -KD * refs.s;
            VectorXd tau_total = tau;

            log << t;
            // q actual
            for(int i=0;i<nv;i++) log << "," << q(i);
            // q deseado
            for(int i=0;i<nv;i++) log << "," << q_des(i);
            // errores
            for(int i=0;i<nv;i++) log << "," << q_err(i);
            // tau_model
            for(int i=0;i<nv;i++) log << "," << tau_model(i);
            // tau_pd
            for(int i=0;i<nv;i++) log << "," << tau_pd(i);
            // tau_total
            for(int i=0;i<nv;i++) log << "," << tau_total(i);
            // masas estimadas
            for(int j=0;j<nv;j++) {
                int idx = j*10;
                double m_est = (idx < real_cols) ? theta_hat(idx) : 0.0;
                log << "," << m_est;
            }
            log << "\n";
        }

        // LOG DETALLADO (Cada 0.5 segundos)
        if (int(t*1000) % 500 == 0) {
            // Calcular Error por Eje
            VectorXd error_per_joint = q - q_des;
            
            cout << "--- t = " << t << " ---" << endl;
            
            // Ver Estado de Articulaciones
            cout << "q_err: " << error_per_joint.transpose().format(IOFormat(4, 0, ", ", " ", "", "", "[", "]")) << endl;
            
            // Ver Torque Generado por el Modelo (Feedforward) vs Feedback (PD)
            VectorXd tau_model = data.jointTorqueRegressor * theta_hat;
            VectorXd tau_pd = -KD * refs.s;
            
            cout << "Tau_Model: " << tau_model.transpose().format(IOFormat(2, 0, ", ", " ", "", "", "[", "]")) << endl;
            cout << "Tau_PD   : " << tau_pd.transpose().format(IOFormat(2, 0, ", ", " ", "", "", "[", "]")) << endl;
            cout << "Tau_Total: " << tau.transpose().format(IOFormat(2, 0, ", ", " ", "", "", "[", "]")) << endl;

            // Ver Parámetros Clave (Masas)
            // Asumiendo estructura de 10 en 10: Indices 0, 10, 20, 30
            cout << "Masas Est: [";
            for(int k=0; k<real_cols; k+=10) cout << theta_hat(k) << (k+10<real_cols ? ", " : "");
            cout << "]" << endl;
            
            cout << endl;
        }
    }

    log.close();
    cout << "Datos guardados en debug_data.csv" << endl;

    return 0;
}