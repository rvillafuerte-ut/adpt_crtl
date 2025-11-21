import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Cargar datos del CSV generado por el controlador
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'output.csv'

df = pd.read_csv(filename)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Control Adaptativo STR - Sistema Lineal', fontsize=14)

# Tracking performance
axes[0, 0].plot(df['t'], df['ref'], 'k--', label='Referencia', linewidth=1.5)
axes[0, 0].plot(df['t'], df['pos'], 'b-', label='Posición', linewidth=1)
axes[0, 0].axvline(x=5.0, color='r', linestyle=':', alpha=0.5, label='Cambio planta')
axes[0, 0].set_ylabel('Posición')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Control signal
axes[0, 1].plot(df['t'], df['u'], 'g-', linewidth=0.8)
axes[0, 1].axvline(x=5.0, color='r', linestyle=':', alpha=0.5)
axes[0, 1].set_ylabel('Señal de Control u')
axes[0, 1].grid(True, alpha=0.3)

# Parámetros estimados
axes[1, 0].plot(df['t'], df['est_a21'], 'b-', linewidth=1)
axes[1, 0].axvline(x=5.0, color='r', linestyle=':', alpha=0.5)
axes[1, 0].axhline(y=-2.0, color='k', linestyle='--', alpha=0.3, label='Inicial: -2.0')
axes[1, 0].axhline(y=-2.0/3.0, color='m', linestyle='--', alpha=0.3, label='Final: -0.67')
axes[1, 0].set_ylabel('Estimado a21')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df['t'], df['est_b2'], 'r-', linewidth=1)
axes[1, 1].axvline(x=5.0, color='r', linestyle=':', alpha=0.5)
axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Inicial: 1.0')
axes[1, 1].axhline(y=1.0/3.0, color='m', linestyle='--', alpha=0.3, label='Final: 0.33')
axes[1, 1].set_ylabel('Estimado b2')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Ganancias del controlador
axes[2, 0].plot(df['t'], df['K1'], 'c-', linewidth=1)
axes[2, 0].axvline(x=5.0, color='r', linestyle=':', alpha=0.5)
axes[2, 0].set_xlabel('Tiempo (s)')
axes[2, 0].set_ylabel('Ganancia K1')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(df['t'], df['K2'], 'm-', linewidth=1)
axes[2, 1].axvline(x=5.0, color='r', linestyle=':', alpha=0.5)
axes[2, 1].set_xlabel('Tiempo (s)')
axes[2, 1].set_ylabel('Ganancia K2')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('str_adaptive_control.png', dpi=150)
print(f"Gráfica guardada: str_adaptive_control.png")
plt.show()
