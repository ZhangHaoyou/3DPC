# pc_fem/visualization/plotting_in_papers.py


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Tuple


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    })

def plot_load_disp_curves_step_load() -> Tuple[plt.Figure, plt.Axes]:

    # --- Load experimental/comparative data ---
    result_folder = 'log/log/B_3500/'
    load_disp = np.loadtxt(result_folder + 'load_disp.csv', delimiter=',')
    exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
    telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)

    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Plot Experiment - black solid
    ax.plot(exp_data[:, 0], exp_data[:, 1], label='Experiment', color='black', linestyle='-', linewidth=1.5)
    
    # Plot Telichko - blue dashed with transparency
    ax.plot(telichko_data[:, 0], telichko_data[:, 1], label="Telichko's Simulation", color=(0, 47/255, 167/255),
            linestyle='--', linewidth=1.0, alpha=0.6)

    # Plot Ours - red dashed with transparency
    ax.plot(-load_disp[:, 0], -load_disp[:, 1], label='This study', color=(192/255, 0, 0),
            linestyle='--', marker='o', linewidth=1.0, alpha=0.6, markersize=4)
    
    # Axis settings
    ax.set_xlabel('Displacement (mm)', fontsize=10)
    ax.set_ylabel('Load (kN)', fontsize=10)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 70)
    ax.set_xticks(np.arange(0, 1.0, 0.2))
    ax.set_yticks(np.arange(0, 80, 10))
    ax.tick_params(labelsize=10)
    
    # Customize tick direction
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))

    # Legend
    ax.legend(fontsize=10, loc='lower right', frameon=False)

    # Save figure
    fig.tight_layout()
    fig.savefig('docs/Figures/Load_disp_across_layers.png', dpi=300)
    
    plt.show()
    
    return fig, ax

if __name__ == '__main__':
    plot_load_disp_curves_step_load()