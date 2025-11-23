import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = 'robot2dof_output.csv'

output_dir = '.'
if len(sys.argv) > 2:
    output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cargar datos
try:
    df = pd.read_csv(fname)
except FileNotFoundError:
    raise SystemExit(f'Archivo no encontrado: {fname}')

# Definir posibles formatos
cols_2dof = {'t','q1','q2','ref1','ref2','tau1','tau2','th1','th2','th3','th4','th5'}
cols_mimo = {'t','ref1','ref2','p1','p2','u1','u2','est_a21','est_b21','est_a43','est_b42'}

mode = None
if cols_2dof.issubset(df.columns):
    mode = '2dof'
elif cols_mimo.issubset(df.columns):
    mode = 'mimo'
else:
    print('Columnas detectadas:', list(df.columns))
    raise SystemExit('Formato desconocido. Ejecute ./2dof_robot_adpt > robot2dof_output.csv para generar el archivo correcto.')

if mode == '2dof':
    # Figura 1: Tracking y torques
    fig1, axes1 = plt.subplots(2, 2, figsize=(10,7))
    fig1.suptitle('Control Adaptativo Slotine-Li - Robot 2DOF')
    axes1[0,0].plot(df['t'], df['ref1'], 'k--', label='q1_ref', lw=1.2)
    axes1[0,0].plot(df['t'], df['q1'], 'b-', label='q1', lw=0.9)
    axes1[0,0].set_ylabel('q1 (rad)')
    axes1[0,0].grid(alpha=0.3)
    axes1[0,0].legend(fontsize=8)
    axes1[0,1].plot(df['t'], df['ref2'], 'k--', label='q2_ref', lw=1.2)
    axes1[0,1].plot(df['t'], df['q2'], 'r-', label='q2', lw=0.9)
    axes1[0,1].set_ylabel('q2 (rad)')
    axes1[0,1].grid(alpha=0.3)
    axes1[0,1].legend(fontsize=8)
    axes1[1,0].plot(df['t'], df['tau1'], 'g-', lw=0.9)
    axes1[1,0].set_ylabel('tau1 (Nm)')
    axes1[1,0].set_xlabel('t (s)')
    axes1[1,0].grid(alpha=0.3)
    axes1[1,1].plot(df['t'], df['tau2'], 'm-', lw=0.9)
    axes1[1,1].set_ylabel('tau2 (Nm)')
    axes1[1,1].set_xlabel('t (s)')
    axes1[1,1].grid(alpha=0.3)
    plt.tight_layout()
    out_path1 = os.path.join(output_dir, 'robot2dof_tracking_torque.png')
    fig1.savefig(out_path1, dpi=150)
    print(f'Guardado: {out_path1}')

    # Figura 2: Parámetros
    fig2, ax2 = plt.subplots(3, 2, figsize=(10,9))
    fig2.suptitle('Evolución Parámetros Estimados (2DOF)')
    params = ['th1','th2','th3','th4','th5']
    true_params = ['th1_true','th2_true','th3_true','th4_true','th5_true']
    labels = ['θ1 Inercia base','θ2 Acople','θ3 Inercia 2','θ4 Gravedad 1','θ5 Gravedad 2']
    have_true = all(tp in df.columns for tp in true_params)
    for i,(p,l,tp) in enumerate(zip(params,labels,true_params)):
        r = i//2
        c = i%2
        ax2[r,c].plot(df['t'], df[p], 'b-', lw=0.9, label='estimado')
        if have_true:
            ax2[r,c].plot(df['t'], df[tp], 'r--', lw=1.0, label='real')
        ax2[r,c].set_ylabel(l)
        ax2[r,c].grid(alpha=0.3)
        ax2[r,c].legend(fontsize=8)
    ax2[2,1].axis('off')
    for a in ax2[2]:
        a.set_xlabel('t (s)')
    plt.tight_layout()
    out_path2 = os.path.join(output_dir, 'robot2dof_params.png')
    fig2.savefig(out_path2, dpi=150)
    print(f'Guardado: {out_path2}')
else:
    # Modo MIMO detectado, graficar básico para informar al usuario
    fig, ax = plt.subplots(3, 2, figsize=(11,9))
    fig.suptitle('Archivo MIMO detectado (mimo_lqr_adpt)')
    ax[0,0].plot(df['t'], df['ref1'], 'k--', label='ref1')
    ax[0,0].plot(df['t'], df['p1'], 'b-', label='p1')
    ax[0,0].set_ylabel('p1'); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)
    ax[0,1].plot(df['t'], df['ref2'], 'k--', label='ref2')
    ax[0,1].plot(df['t'], df['p2'], 'r-', label='p2')
    ax[0,1].set_ylabel('p2'); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)
    ax[1,0].plot(df['t'], df['u1'], 'g-', label='u1')
    ax[1,0].set_ylabel('u1'); ax[1,0].grid(alpha=0.3)
    ax[1,1].plot(df['t'], df['u2'], 'm-', label='u2')
    ax[1,1].set_ylabel('u2'); ax[1,1].grid(alpha=0.3)
    ax[2,0].plot(df['t'], df['est_a21'], 'c-', label='a21')
    ax[2,0].plot(df['t'], df['est_b21'], 'y-', label='b21')
    ax[2,0].set_ylabel('Fila 2'); ax[2,0].legend(); ax[2,0].grid(alpha=0.3)
    ax[2,1].plot(df['t'], df['est_a43'], 'c-', label='a43')
    ax[2,1].plot(df['t'], df['est_b42'], 'y-', label='b42')
    ax[2,1].set_ylabel('Fila 4'); ax[2,1].legend(); ax[2,1].grid(alpha=0.3)
    for a in ax[2]:
        a.set_xlabel('t (s)')
    plt.tight_layout()
    out_path_mimo = os.path.join(output_dir, 'mimo_detected.png')
    fig.savefig(out_path_mimo, dpi=150)
    print(f'Guardado: {out_path_mimo}')
    print('NOTA: Para generar robot2dof_output.csv correcto ejecute: ./2dof_robot_adpt > robot2dof_output.csv')

# plt.show()  # Comentado para evitar warning en modo no-interactivo