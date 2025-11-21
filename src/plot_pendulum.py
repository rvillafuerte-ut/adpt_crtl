import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'pendulum_output.csv'

df = pd.read_csv(filename)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Control Adaptativo STR - Péndulo No Lineal', fontsize=14)

# Tracking
axes[0, 0].plot(df['t'], df['ref'], 'k--', label='Referencia', linewidth=1.5)
axes[0, 0].plot(df['t'], df['theta'], 'b-', label='θ', linewidth=1)
axes[0, 0].axvline(x=8.0, color='r', linestyle=':', alpha=0.5, label='Cambio planta')
axes[0, 0].set_ylabel('Ángulo θ (rad)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Control
axes[0, 1].plot(df['t'], df['u'], 'g-', linewidth=0.8)
axes[0, 1].axvline(x=8.0, color='r', linestyle=':', alpha=0.5)
axes[0, 1].set_ylabel('Torque u (Nm)')
axes[0, 1].grid(True, alpha=0.3)

# Parámetro lineal
axes[1, 0].plot(df['t'], df['est_a21_lin'], 'b-', linewidth=1)
axes[1, 0].axvline(x=8.0, color='r', linestyle=':', alpha=0.5)
axes[1, 0].set_ylabel('Estimado a21_lin')
axes[1, 0].grid(True, alpha=0.3)

# Parámetro no lineal (gravedad)
axes[1, 1].plot(df['t'], df['est_a21_nl'], 'r-', linewidth=1)
axes[1, 1].axvline(x=8.0, color='r', linestyle=':', alpha=0.5)
axes[1, 1].axhline(y=-9.81, color='k', linestyle='--', alpha=0.3, label='Real: -g/L=-9.81')
axes[1, 1].set_ylabel('Estimado a21_nl')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Estimado b2
axes[2, 0].plot(df['t'], df['est_b2'], 'c-', linewidth=1)
axes[2, 0].axvline(x=8.0, color='r', linestyle=':', alpha=0.5)
axes[2, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Inicial: 1.0')
axes[2, 0].axhline(y=0.4, color='m', linestyle='--', alpha=0.3, label='Final: 0.4')
axes[2, 0].set_xlabel('Tiempo (s)')
axes[2, 0].set_ylabel('Estimado b2')
axes[2, 0].legend(fontsize=8)
axes[2, 0].grid(True, alpha=0.3)

# Ganancias
axes[2, 1].plot(df['t'], df['K1'], 'c-', linewidth=1, label='K1')
axes[2, 1].plot(df['t'], df['K2'], 'm-', linewidth=1, label='K2')
axes[2, 1].axvline(x=8.0, color='r', linestyle=':', alpha=0.5)
axes[2, 1].set_xlabel('Tiempo (s)')
axes[2, 1].set_ylabel('Ganancias')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pendulum_adaptive_control.png', dpi=150)
print(f"Gráfica guardada: pendulum_adaptive_control.png")
plt.show()
