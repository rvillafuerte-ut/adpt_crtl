#!/usr/bin/env python3
"""
Analiza 'chattering' (oscilación rápida) en las señales de torque tau_total.
Calcula:
- Derivada aproximada (dif / dt_log)
- Métricas: max |dTau|, RMS dTau, porcentaje de muestras con |dTau| > umbral
- Sugerencias de mitigación
"""
import pandas as pd, numpy as np, os
CSV_INPUT = "../src/build/debug_data.csv"
print(f"Leyendo {CSV_INPUT}")
try:
    data = pd.read_csv(CSV_INPUT)
except FileNotFoundError:
    print("No se encontró debug_data.csv. Ejecuta primero el binario.")
    exit(1)
# Detect dt del log (asumimos paso constante)
if len(data) < 3:
    print("Datos insuficientes para análisis.")
    exit(0)
# Paso de log ~ cada 0.01 s (10ms) por diseño
dt_log = (data['t'].iloc[1] - data['t'].iloc[0])
print(f"dt_log estimado: {dt_log:.6f} s")
# Columnas tau_total*
tau_cols = [c for c in data.columns if c.startswith('tau_total')]
report = []
for c in tau_cols:
    tau = data[c].values
    d_tau = np.diff(tau) / dt_log  # derivada discreta
    abs_d = np.abs(d_tau)
    max_abs = abs_d.max()
    rms = np.sqrt((d_tau**2).mean())
    thr = 50.0  # umbral arbitrario Nm/s para identificar chattering fuerte
    pct = (abs_d > thr).mean() * 100.0
    report.append((c,max_abs,rms,pct))
print("\nMÉTRICAS DE CHATTERING (derivada torque):")
print("Columna, Max|dTau| (Nm/s), RMS dTau (Nm/s), %>|50|")
for r in report:
    print(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.1f}%")
# Sugerencias básicas (no automatizadas):
print("\nSUGERENCIAS DE MITIGACIÓN:")
print("1. Aumentar KD gradualmente (0.10 -> 0.5) si error grande pero chattering bajo; si chattering alto, reducir Gamma.")
print("2. Reducir Gamma (0.5 -> 0.1) y/o añadir filtro exponencial: theta_hat = alpha*theta_hat_prev + (1-alpha)*theta_hat_new, alpha≈0.9.")
print("3. Implementar deadzone adaptación: si |s_i| < s_thr (p.e. 0.02) no actualizar ese parámetro.")
print("4. Limitar variación de tau: tau = tau_prev + clip(tau - tau_prev, -dLim*dt, dLim*dt) con dLim≈50 Nm/s.")
print("5. Normalizar columnas del regresor: Y_norm = Y * diag(1/(RMS_col+eps)) para evitar saturaciones.")
print("6. Añadir término -K_s * s directo (ya presente KD) y opcional filtro pasa-bajo en tau_model: tau_model_f = beta*tau_model_prev + (1-beta)*tau_model, beta≈0.95.")
