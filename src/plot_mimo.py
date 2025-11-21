import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'mimo_output.csv'

df = pd.read_csv(filename)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Control Adaptativo LQR - Sistema MIMO (4 estados)', fontsize=14)

# Tracking masa 1
axes[0, 0].plot(df['t'], df['ref1'], 'k--', label='Ref1', linewidth=1.5)
axes[0, 0].plot(df['t'], df['p1'], 'b-', label='p1', linewidth=1)
axes[0, 0].axvline(x=6.0, color='r', linestyle=':', alpha=0.5, label='Cambio')
axes[0, 0].set_ylabel('Posición p1')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Tracking masa 2
axes[0, 1].plot(df['t'], df['ref2'], 'k--', label='Ref2', linewidth=1.5)
axes[0, 1].plot(df['t'], df['p2'], 'r-', label='p2', linewidth=1)
axes[0, 1].axvline(x=6.0, color='r', linestyle=':', alpha=0.5)
axes[0, 1].set_ylabel('Posición p2')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Señales de control
axes[1, 0].plot(df['t'], df['u1'], 'g-', linewidth=0.8, label='u1')
axes[1, 0].axvline(x=6.0, color='r', linestyle=':', alpha=0.5)
axes[1, 0].set_ylabel('Control u1')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df['t'], df['u2'], 'm-', linewidth=0.8, label='u2')
axes[1, 1].axvline(x=6.0, color='r', linestyle=':', alpha=0.5)
axes[1, 1].set_ylabel('Control u2')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Parámetros estimados - Estimador 1
axes[2, 0].plot(df['t'], df['est_a21'], 'b-', linewidth=1, label='a21')
axes[2, 0].plot(df['t'], df['est_b21'], 'c-', linewidth=1, label='b21')
axes[2, 0].axvline(x=6.0, color='r', linestyle=':', alpha=0.5)
axes[2, 0].set_xlabel('Tiempo (s)')
axes[2, 0].set_ylabel('Estimados (Fila 2)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Parámetros estimados - Estimador 2
axes[2, 1].plot(df['t'], df['est_a43'], 'r-', linewidth=1, label='a43')
axes[2, 1].plot(df['t'], df['est_b42'], 'm-', linewidth=1, label='b42')
axes[2, 1].axvline(x=6.0, color='r', linestyle=':', alpha=0.5)
axes[2, 1].set_xlabel('Tiempo (s)')
axes[2, 1].set_ylabel('Estimados (Fila 4)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mimo_lqr_adaptive.png', dpi=150)
print(f"Gráfica guardada: mimo_lqr_adaptive.png")
plt.show()
