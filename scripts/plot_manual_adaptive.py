#!/usr/bin/env python3
"""
Script para plotear resultados del controlador adaptativo manual (4dof_robot_adpt_manual)
Los archivos de entrada y salida están hardcodeados para facilitar el uso.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================
# CONFIGURACIÓN DE RUTAS (HARDCODED)
# ============================================
CSV_INPUT = "../src/build/debug_data.csv"
RESULTS_DIR = "../results/"

# Crear directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================
# CARGAR DATOS
# ============================================
print(f"Cargando datos desde: {CSV_INPUT}")
try:
    data = pd.read_csv(CSV_INPUT)
    print(f"✓ Datos cargados: {len(data)} muestras")
    print(f"✓ Columnas: {list(data.columns)}")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo {CSV_INPUT}")
    print("Asegúrate de haber ejecutado './4dof_robot_adpt_manual' primero")
    exit(1)

# ============================================
# CREAR FIGURA: 3 filas x NV columnas (error / ley de control / adaptación)
# ============================================
# detectar NV a partir de columnas q_err*
q_err_cols = [c for c in data.columns if c.startswith('q_err')]
if len(q_err_cols) == 0:
    print('ERROR: No se encontraron columnas q_err*. Asegúrate de que el ejecutable haya creado debug_data.csv con el nuevo formato.')
    exit(1)
nv = len(q_err_cols)

fig, axes = plt.subplots(3, nv, figsize=(4*nv, 10), sharex=True)
fig.suptitle('Control Adaptativo Manual — 4DOF: Error / Ley de Control / Adaptación', fontsize=14)

for j in range(nv):
    # Row 0: error por articulación
    ax_err = axes[0, j] if nv>1 else axes[0]
    col_err = f'q_err{j+1}'
    ax_err.plot(data['t'], data[col_err], color='C0')
    ax_err.axhline(0, color='k', linestyle='--', linewidth=0.6)
    ax_err.set_title(f'Joint {j+1} — Error')
    ax_err.grid(True, alpha=0.3)

    # Row 1: Ley de control (model, PD, total)
    ax_ctrl = axes[1, j] if nv>1 else axes[1]
    col_m = f'tau_model{j+1}'
    col_pd = f'tau_pd{j+1}'
    col_tot = f'tau_total{j+1}'
    ax_ctrl.plot(data['t'], data[col_m], label='Model', color='C1')
    ax_ctrl.plot(data['t'], data[col_pd], label='PD', color='C2')
    ax_ctrl.plot(data['t'], data[col_tot], label='Total', color='C3', alpha=0.8)
    ax_ctrl.set_title(f'Joint {j+1} — Torque (Nm)')
    ax_ctrl.grid(True, alpha=0.3)
    if j == 0:
        ax_ctrl.legend(fontsize=8)

    # Row 2: Adaptación (masa estimada)
    ax_adapt = axes[2, j] if nv>1 else axes[2]
    col_mass = f'mass_est{j+1}'
    if col_mass in data.columns:
        ax_adapt.plot(data['t'], data[col_mass], color='C4')
    ax_adapt.set_title(f'Joint {j+1} — Mass est.')
    ax_adapt.grid(True, alpha=0.3)

for ax in (axes[2, :] if nv>1 else [axes[2]]):
    ax.set_xlabel('Tiempo [s]')

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_file = os.path.join(RESULTS_DIR, "manual_adaptive_results.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Gráfica guardada en: {output_file}")
plt.close()

# Estadísticas por articulación (máximo error)
print('\n' + '='*60)
print('RESUMEN POR ARTICULACIÓN')
print('='*60)
for j in range(nv):
    col_err = f'q_err{j+1}'
    err_max = data[col_err].abs().max()
    print(f'Joint {j+1}: Max err = {err_max:.4f} rad ({np.rad2deg(err_max):.2f}°)')
print('='*60)
