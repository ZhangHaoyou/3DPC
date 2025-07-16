# pc_fem/visualization/plotting.py


import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


from pc_fem.config_loader import load_config


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    })

conc_prop, dmg_para, geo_para, mesh_para, contact_para = load_config()

def plot_load_disp_curves_step_load() -> Tuple[plt.Figure, plt.Axes]:

    # --- Load experimental/comparative data ---
    load_disp_tributary = np.loadtxt('log/log/tributary_50000_Bbar/load_disp.csv', delimiter=',')
    load_disp_tributary_025 = np.loadtxt('log/log/tributary_50000_Bbar_0.25/load_disp.csv', delimiter=',')
    load_disp_tributary_1 = np.loadtxt('log/log/tributary_50000_Bbar_1/load_disp.csv', delimiter=',')
    exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
    telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)

    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plot Experiment - black solid
    ax.plot(exp_data[:, 0], exp_data[:, 1], label='Experiment', color='black', linestyle='-', linewidth=2)
    
    # Plot Telichko - blue dashed with transparency
    ax.plot(telichko_data[:, 0], telichko_data[:, 1], label="Telichko's Simulation", color=(0, 47/255, 167/255),
            linestyle='--', linewidth=1.5, alpha=0.6)

    # Plot Ours - red dashed with transparency
    ax.plot(-load_disp_tributary_1[:, 0], -load_disp_tributary_1[:, 1], label='1 MPa per step', color=(0, 47/255, 167/255),
            linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
    
    ax.plot(-load_disp_tributary[:, 0], -load_disp_tributary[:, 1], label='0.5 MPa per step', color=(192/255, 0, 0),
            linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
    
    ax.plot(-load_disp_tributary_025[:, 0], -load_disp_tributary_025[:, 1], label='0.25 MPa per step', color='green',
            linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
    
    # Add parameter info in the bottom right corner
    param_text = (f"$A_{{\\mathrm{{c}}}}$ = {dmg_para.Ac:.2f}, $B_{{\\mathrm{{c}}}}$ = {dmg_para.Bc:.0f}\n"
            f"$A_{{\\mathrm{{t}}}}$ = {dmg_para.At:.2f}, $B_{{\\mathrm{{t}}}}$ = {dmg_para.Bt:.0f}\n"
            rf"$\epsilon_N$ = {contact_para.epsilon[0]}, $\epsilon_T$ = {contact_para.epsilon[1]}")
    ax.text(0.97, 0.05, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right')

    # Axis settings
    ax.set_xlabel('Displacement (mm)', fontsize=16)
    ax.set_ylabel('Load (kN)', fontsize=16)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 70)
    ax.set_xticks(np.arange(0, 0.35, 0.05))
    ax.set_yticks(np.arange(0, 80, 10))
    ax.tick_params(labelsize=12)

    # Legend
    ax.legend(fontsize=14, loc='upper left')

    # Save figure
    fig.tight_layout()
    fig.savefig('log/log/step_load.png', dpi=300)
    
    plt.show()
    
    return fig, ax

def plot_load_disp_curves_loads() -> Tuple[plt.Figure, plt.Axes]:

    # --- Load experimental/comparative data ---
    load_disp_mean = np.loadtxt('log/log/mean_50000_Bbar/load_disp.csv', delimiter=',')
    load_disp_tributary = np.loadtxt('log/log/tributary_50000_Bbar/load_disp.csv', delimiter=',')
    exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
    telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)

    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plot Experiment - black solid
    ax.plot(exp_data[:, 0], exp_data[:, 1], label='Experiment', color='black', linestyle='-', linewidth=2)
    
    # Plot Telichko - blue dashed with transparency
    ax.plot(telichko_data[:, 0], telichko_data[:, 1], label="Telichko's Simulation", color=(0, 47/255, 167/255),
            linestyle='--', linewidth=1.5, alpha=0.6)

    # Plot Ours - red dashed with transparency
    ax.plot(-load_disp_mean[:, 0], -load_disp_mean[:, 1], label='Mean load', color=(0, 47/255, 167/255),
            linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
    
    ax.plot(-load_disp_tributary[:, 0], -load_disp_tributary[:, 1], label='Tributary load', color=(192/255, 0, 0),
            linestyle='--', marker='o', linewidth=1.5, alpha=0.6)
    
    # Add parameter info in the bottom right corner
    param_text = (f"$A_{{\\mathrm{{c}}}}$ = {dmg_para.Ac:.2f}, $B_{{\\mathrm{{c}}}}$ = {dmg_para.Bc:.0f}\n"
            f"$A_{{\\mathrm{{t}}}}$ = {dmg_para.At:.2f}, $B_{{\\mathrm{{t}}}}$ = {dmg_para.Bt:.0f}\n"
            rf"$\epsilon_N$ = {contact_para.epsilon[0]}, $\epsilon_T$ = {contact_para.epsilon[1]}")
    ax.text(0.97, 0.05, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right')

    # Axis settings
    ax.set_xlabel('Displacement (mm)', fontsize=16)
    ax.set_ylabel('Load (kN)', fontsize=16)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 70)
    ax.set_xticks(np.arange(0, 0.35, 0.05))
    ax.set_yticks(np.arange(0, 80, 10))
    ax.tick_params(labelsize=12)

    # Legend
    ax.legend(fontsize=14, loc='upper left')

    # Save figure
    fig.tight_layout()
    fig.savefig('log/log/mean_tributary.png', dpi=300)
    
    plt.show()
    
    return fig, ax


if __name__ == '__main__':
    plot_load_disp_curves_step_load()
#     plot_load_disp_curves_loads()
    pass