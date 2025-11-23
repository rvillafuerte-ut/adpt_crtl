#!/usr/bin/env python3
"""
Genera PNG mostrando seguimiento de trayectoria: q vs q_des y errores.
Lee debug_data.csv (hardcodeado) generado por 4dof_robot_adpt_manual.
"""
import pandas as pd, os, matplotlib.pyplot as plt, numpy as np
CSV_INPUT = "../src/build/debug_data.csv"
RESULTS_DIR = "../results/"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Leyendo {CSV_INPUT}")
data = pd.read_csv(CSV_INPUT)
# Detectar número de articulaciones por columnas qN
q_cols = [c for c in data.columns if c.startswith('q') and c[1].isdigit() and not c.startswith('q_des') and not c.startswith('q_err')]
# Filtrar exactamente 'q1','q2',... evitando 'q_des1'
q_cols = [c for c in q_cols if c.startswith('q') and not c.startswith('q_des') and not c.startswith('q_err')]
# Ordenar por índice
q_cols = sorted(q_cols, key=lambda x:int(x[1:]))
nv = len(q_cols)
# Preparar figura: dos filas (q seguimiento y error) por joint => 2*nv subplots en una sola columna o en grid
fig, axes = plt.subplots(nv, 2, figsize=(12, 3*nv), sharex=True)
if nv == 1: axes = np.array([axes])
for j in range(nv):
    q_col = f"q{j+1}"
    qd_col = f"q_des{j+1}"
    err_col = f"q_err{j+1}"
    ax_trk = axes[j,0]
    ax_err = axes[j,1]
    ax_trk.plot(data['t'], data[qd_col], label='q_des', color='C1')
    ax_trk.plot(data['t'], data[q_col], label='q', color='C0', alpha=0.8)
    ax_trk.set_ylabel(f'Joint {j+1} [rad]')
    ax_trk.set_title(f'Seguimiento Joint {j+1}')
    ax_trk.grid(alpha=0.3)
    if j==0: ax_trk.legend(loc='upper right', fontsize=8)
    ax_err.plot(data['t'], data[err_col], color='C3')
    ax_err.axhline(0, color='k', lw=0.6, ls='--')
    ax_err.set_title(f'Error Joint {j+1}')
    ax_err.grid(alpha=0.3)
    # Métricas
    e_abs = data[err_col].abs()
    txt = f"Max: {e_abs.max():.3f}\nRMS: {np.sqrt((e_abs**2).mean()):.3f}"
    ax_err.text(0.98,0.95,txt,transform=ax_err.transAxes,ha='right',va='top',fontsize=8,
                bbox=dict(boxstyle='round',facecolor='white',alpha=0.6))
axes[-1,0].set_xlabel('Tiempo [s]')
axes[-1,1].set_xlabel('Tiempo [s]')
fig.suptitle('Seguimiento de Trayectoria y Error por Articulación', fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
outfile = os.path.join(RESULTS_DIR,'manual_adaptive_tracking.png')
fig.savefig(outfile,dpi=150)
print(f"PNG guardado: {outfile}")
