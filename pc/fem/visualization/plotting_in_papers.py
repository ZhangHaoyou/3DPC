# fem/visualization/plotting_in_papers.py


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
    result_folder = 'log/fem/'
    load_disp = np.loadtxt(result_folder + 'load_disp.csv', delimiter=',')
    exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
    telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Plot Experiment - black solid
    ax.plot(exp_data[:, 0], exp_data[:, 1], label='Experiment', color='black', linestyle='-', linewidth=3.5)
    exp_disp, exp_load = exp_data[-1, :]
    ax.scatter(exp_disp, exp_load, marker='o', s=48, facecolors='white', edgecolors='black',linewidth=2, zorder=5)
    ax.annotate(f"({exp_disp:.2f}, {exp_load:.1f})",
            (exp_disp, exp_load),            # point coordinates
            xytext=(-20, -13),               # offset in pixels
            textcoords='offset points',
            fontsize=10,
            color='black')
    
    # Plot Telichko - blue dashed with transparency
    ax.plot(telichko_data[:, 0], telichko_data[:, 1], label="Telichko's Simulation", color=(0, 47/255, 167/255),
            linestyle='--', linewidth=2.5)
    telichko_disp, telichko_load = telichko_data[-1, :]
    ax.scatter(telichko_disp, telichko_load, marker='o', s=48, facecolors='white', edgecolors=(0, 47/255, 167/255),
            linewidth=2, zorder=5)
    ax.annotate(f"({telichko_disp:.2f}, {telichko_load:.1f})",
            (telichko_disp, telichko_load),  # point coordinates
            xytext=(-20, 10),               # offset in pixels
            textcoords='offset points',
            fontsize=10,
            color=(0, 47/255, 167/255))
    
    # Plot Ours - red dashed with transparency
    ax.plot(-load_disp[:, 0], -load_disp[:, 1], label='This study', color=(192/255, 0, 0),
            linestyle='-',  linewidth=2.5)
    ours_disp, ours_load = -load_disp[-1, :]
    ax.scatter(ours_disp, ours_load, marker='o', s=48, facecolors='white', edgecolors=(192/255, 0, 0), linewidth=2, zorder=5)
    ax.annotate(f"({ours_disp:.2f}, {ours_load:.1f})",
            (ours_disp, ours_load),          # point coordinates
            xytext=(-33, -15),               # offset in pixels
            textcoords='offset points',
            fontsize=10,
            color=(192/255, 0, 0))
    
    # Axis settings
    ax.set_xlabel('Displacement (mm)', fontsize=11)
    ax.set_ylabel('Load (kN)', fontsize=11)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 80)
    ax.set_xticks(np.arange(0, 0.4, 0.1))
    ax.set_yticks(np.arange(0, 90, 20))
    ax.tick_params(labelsize=10)
    
    # Customize tick direction
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    
    # Legend
    ax.legend(fontsize=10, loc='upper left', frameon=False)
    
    # Save figure
    fig.tight_layout()
    fig.savefig(result_folder + 'Load_disp_across_layers.pdf', dpi=300)
    
    return fig, ax

def plot_training_process_mlp_pde() -> Tuple[plt.Figure, plt.Axes]:
    result_folder = 'log/pinn/'
    # neuron16 = np.loadtxt(result_folder + '16x5/loss/layer0.csv', delimiter=',')
    neuron8 = np.loadtxt(result_folder + '8x5/loss/layer0.csv', delimiter=',')
    neuron4 = np.loadtxt(result_folder + '4x5/loss/layer0.csv', delimiter=',')
    neuron3 = np.loadtxt(result_folder + '3x5/loss/layer0.csv', delimiter=',')
    neuron2 = np.loadtxt(result_folder + '2x5/loss/layer0.csv', delimiter=',')
    # neuron = np.loadtxt(result_folder + 'loss/layer0.csv', delimiter=',')
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    loss_column = 1
    # ax.plot(list(range(1, len(neuron16)+1)), neuron16[:,loss_column], label='16 neurons', color='black', linewidth=1.5)
    ax.plot(list(range(1, len(neuron8)+1)), neuron8[:,loss_column], label='8 neurons',
            color='forestgreen', linewidth=2.5)
    ax.plot(list(range(1, len(neuron4)+1)), neuron4[:,loss_column], label='4 neurons',
            color=(192/255, 0, 0), linewidth=2.5)
    ax.plot(list(range(1, len(neuron3)+1)), neuron3[:,loss_column], label='3 neurons',
            color='purple', linewidth=2.5)
    ax.plot(list(range(1, len(neuron2)+1)), neuron2[:,loss_column], label='2 neurons',
            color=(0, 47/255, 167/255), linewidth=2.5)
    # ax.plot(list(range(1, len(neuron)+1)), neuron[:,loss_column], label='Current', color='blue', linewidth=1.5)
    
    # ax.plot([1000.0, 1000.0], [1.0, 0.0], color='black', linewidth=0.5)
    # ax.arrow(1000.0, 0.1, -300.0, 0.0, head_width=0.025, head_length=40.0, fc='black', ec='black')
    # ax.arrow(1000.0, 0.1,  300.0, 0.0, head_width=0.025, head_length=40.0, fc='black', ec='black')
    # ax.text(850.0, 0.2, "Adam", ha='center', fontsize=10)
    # ax.text(1150.0, 0.2, "LBFGS", ha='center', fontsize=10)
    
    ax.set_xlabel('Number of epochs', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_yscale('log')
    ax.set_xlim(0, 30000)
    ax.set_ylim(1.0e-6, 1.0e0)
    ax.set_xticks(np.arange(0, 35000, 10000))
    ax.tick_params(labelsize=10)
    
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    
    ax.legend(fontsize=10, ncol=2, loc='upper center', frameon=False)
    
    fig.tight_layout()
    fig.savefig(result_folder + 'training_process_mlp_pde.pdf', dpi=300)
    
    return fig, ax

def plot_training_process_mlp_bcs() -> Tuple[plt.Figure, plt.Axes]:
    result_folder = 'log/pinn/'
    # neuron16 = np.loadtxt(result_folder + '16x5/loss/layer0.csv', delimiter=',')
    neuron8 = np.loadtxt(result_folder + '8x5/loss/layer0.csv', delimiter=',')
    neuron4 = np.loadtxt(result_folder + '4x5/loss/layer0.csv', delimiter=',')
    neuron3 = np.loadtxt(result_folder + '3x5/loss/layer0.csv', delimiter=',')
    neuron2 = np.loadtxt(result_folder + '2x5/loss/layer0.csv', delimiter=',')
    # neuron = np.loadtxt(result_folder + 'loss/layer0.csv', delimiter=',')
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    loss_column = 2
    # ax.plot(list(range(1, len(neuron16)+1)), neuron16[:,loss_column], label='16 neurons', color='black', linewidth=1.5)
    ax.plot(list(range(1, len(neuron8)+1)), neuron8[:,loss_column], label='8 neurons',
            color='forestgreen', linewidth=2.5)
    ax.plot(list(range(1, len(neuron4)+1)), neuron4[:,loss_column], label='4 neurons',
            color=(192/255, 0, 0), linewidth=2.5)
    ax.plot(list(range(1, len(neuron3)+1)), neuron3[:,loss_column], label='3 neurons',
            color='purple', linewidth=2.5)
    ax.plot(list(range(1, len(neuron2)+1)), neuron2[:,loss_column], label='2 neurons',
            color=(0, 47/255, 167/255), linewidth=2.5)
    # ax.plot(list(range(1, len(neuron)+1)), neuron[:,loss_column], label='Current', color='green', linewidth=1.5)
    
    # ax.plot([1000.0, 1000.0], [1.0, 0.0], color='black', linewidth=0.5)
    # ax.arrow(1000.0, 0.1, -300.0, 0.0, head_width=0.025, head_length=40.0, fc='black', ec='black')
    # ax.arrow(1000.0, 0.1,  300.0, 0.0, head_width=0.025, head_length=40.0, fc='black', ec='black')
    # ax.text(850.0, 0.2, "Adam", ha='center', fontsize=10)
    # ax.text(1150.0, 0.2, "LBFGS", ha='center', fontsize=10)
    
    ax.set_xlabel('Number of epochs', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_yscale('log')
    ax.set_xlim(0, 30000)
    ax.set_ylim(1.0e-6, 1.0e1)
    ax.set_xticks(np.arange(0, 35000, 10000))
    ax.tick_params(labelsize=10)
    
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    
    ax.legend(fontsize=10, ncol=2, loc='upper center', frameon=False)
    
    fig.tight_layout()
    fig.savefig(result_folder + 'training_process_mlp_bcs.pdf', dpi=300)
    
    return fig, ax

def plot_training_process_graph_pde() -> Tuple[plt.Figure, plt.Axes]:
    result_folder = 'log/graph/'
    # neuron16 = np.loadtxt(result_folder + '16x2/loss/layer0.csv', delimiter=',')
    # neuron12 = np.loadtxt(result_folder + '12x2/loss/layer0.csv', delimiter=',')
    neuron8 = np.loadtxt(result_folder + '8x2/loss/layer0.csv', delimiter=',')
    neuron4 = np.loadtxt(result_folder + '4x2/loss/layer0.csv', delimiter=',')
    neuron3 = np.loadtxt(result_folder + '3x2/loss/layer0.csv', delimiter=',')
    neuron2 = np.loadtxt(result_folder + '2x2/loss/layer0.csv', delimiter=',')
    # neuron = np.loadtxt(result_folder + 'loss/layer0.csv', delimiter=',')
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    loss_column = 1
    # ax.plot(list(range(1, len(neuron16)+1)), neuron16[:,loss_column], label='16 dimensions', color='black', linewidth=1.5)
    # ax.plot(list(range(1, len(neuron12)+1)), neuron12[:,loss_column], label='12 dimensions', color='gray', linewidth=1.5)
    ax.plot(list(range(1, len(neuron2)+1)), neuron2[:,loss_column], label='2 dimensions',
            color=(0, 47/255, 167/255), linewidth=2.5)
    ax.plot(list(range(1, len(neuron3)+1)), neuron3[:,loss_column], label='3 dimensions',
            color=(192/255, 0, 0), linewidth=2.5)
    ax.plot(list(range(1, len(neuron4)+1)), neuron4[:,loss_column], label='4 dimensions',
            color=(255/255, 204/255, 0), linewidth=2.5)
    ax.plot(list(range(1, len(neuron8)+1)), neuron8[:,loss_column], label='8 dimensions',
            color='forestgreen', linewidth=2.5)
    # ax.plot(list(range(1, len(neuron)+1)), neuron[:,loss_column], label='Current', color='purple', linewidth=1.5)
    
    # ax.plot([1000.0, 1000.0], [1.0, 0.0], color='black', linewidth=0.5)
    # ax.arrow(1000.0, 0.1, -300.0, 0.0, head_width=0.025, head_length=40.0, fc='black', ec='black')
    # ax.arrow(1000.0, 0.1,  300.0, 0.0, head_width=0.025, head_length=40.0, fc='black', ec='black')
    # ax.text(850.0, 0.2, "Adam", ha='center', fontsize=10)
    # ax.text(1150.0, 0.2, "LBFGS", ha='center', fontsize=10)
    
    ax.set_xlabel('Number of epochs', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_yscale('log')
    ax.set_xlim(0, 30000)
    ax.set_ylim(1.0e-6, 1.0e4)
    ax.set_xticks(np.arange(0, 35000, 10000))
    ax.tick_params(labelsize=10)
    
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    
    ax.legend(fontsize=10, ncol=2, loc='upper center', frameon=False)
    
    fig.tight_layout()
    fig.savefig(result_folder + 'training_process_graph_pde.pdf', dpi=300)
    
    return fig, ax

def plot_training_process_graph_bcs() -> Tuple[plt.Figure, plt.Axes]:
    result_folder = 'log/graph/'
    # neuron16 = np.loadtxt(result_folder + '16x2/loss/layer0.csv', delimiter=',')
    # neuron12 = np.loadtxt(result_folder + '12x2/loss/layer0.csv', delimiter=',')
    neuron8 = np.loadtxt(result_folder + '8x2/loss/layer0.csv', delimiter=',')
    neuron4 = np.loadtxt(result_folder + '4x2/loss/layer0.csv', delimiter=',')
    neuron3 = np.loadtxt(result_folder + '3x2/loss/layer0.csv', delimiter=',')
    neuron2 = np.loadtxt(result_folder + '2x2/loss/layer0.csv', delimiter=',')
    # neuron = np.loadtxt(result_folder + 'loss/layer0.csv', delimiter=',')
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    loss_column = 2
    # ax.plot(list(range(1, len(neuron16)+1)), neuron16[:,loss_column], label='16 dimensions', color='black', linewidth=1.5)
    # ax.plot(list(range(1, len(neuron12)+1)), neuron12[:,loss_column], label='12 dimensions', color='gray', linewidth=1.5)
    ax.plot(list(range(1, len(neuron2)+1)), neuron2[:,loss_column], label='2 dimensions',
            color=(0, 47/255, 167/255), linewidth=2.5)
    ax.plot(list(range(1, len(neuron3)+1)), neuron3[:,loss_column], label='3 dimensions',
            color=(192/255, 0, 0), linewidth=2.5)
    ax.plot(list(range(1, len(neuron4)+1)), neuron4[:,loss_column], label='4 dimensions',
            color=(255/255, 204/255, 0), linewidth=2.5)
    ax.plot(list(range(1, len(neuron8)+1)), neuron8[:,loss_column], label='8 dimensions',
            color='forestgreen', linewidth=2.5)
    # ax.plot(list(range(1, len(neuron)+1)), neuron[:,loss_column], label='Current', color='purple', linewidth=1.5)
    
    # ax.plot([1000.0, 1000.0], [1.0, 0.0], color='black', linewidth=0.5)
    # ax.arrow(1000.0, 0.1, -300.0, 0.0, head_width=0.025, head_length=40.0, fc='black', ec='black')
    # ax.arrow(1000.0, 0.1,  300.0, 0.0, head_width=0.025, head_length=40.0, fc='black', ec='black')
    # ax.text(850.0, 0.2, "Adam", ha='center', fontsize=10)
    # ax.text(1150.0, 0.2, "LBFGS", ha='center', fontsize=10)
    
    ax.set_xlabel('Number of epochs', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_yscale('log')
    ax.set_xlim(0, 30000)
    ax.set_ylim(1.0e-6, 1.0e4)
    ax.set_xticks(np.arange(0, 35000, 10000))
    ax.tick_params(labelsize=10)
    
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    
    ax.legend(fontsize=10, ncol=2, loc='upper center', frameon=False)
    
    fig.tight_layout()
    fig.savefig(result_folder + 'training_process_graph_bcs.pdf', dpi=300)
    
    return fig, ax

def plot_training_process_total_loss() -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    loss_column = 3
    
    mlp_folder = 'log/pinn/'
    # neuron16 = np.loadtxt(mlp_folder  + '16x5/loss/layer0.csv', delimiter=',')
    # neuron8 = np.loadtxt(mlp_folder  + '8x5/loss/layer0.csv', delimiter=',')
    neuron4 = np.loadtxt(mlp_folder  + '4x5/loss/layer0.csv', delimiter=',')
    # neuron3 = np.loadtxt(mlp_folder  + '3x5/loss/layer0.csv', delimiter=',')
    
    # ax.plot(list(range(1, len(neuron16)+1)), neuron16[:,loss_column], label='16 neurons', color='black', linewidth=1.5)
    # ax.plot(list(range(1, len(neuron8)+1)), neuron8[:,loss_column], label='8 neurons', color='purple', linewidth=1.5)
    ax.plot(list(range(1, len(neuron4)+1)), neuron4[:,loss_column], label='4 neurons',
            color=(0, 47/255, 167/255), linewidth=2.5)
    # ax.plot(list(range(1, len(neuron3)+1)), neuron3[:,loss_column], label='3 neurons', color='blue', linewidth=1.5)
    
    graph_folder = 'log/graph/'
    dim3 = np.loadtxt(graph_folder + '3x2/loss/layer0.csv', delimiter=',')
    ax.plot(list(range(1, len(dim3)+1)), dim3[:,loss_column], label='3 dimensions',
            color=(192/255, 0, 0), linewidth=2.5)
    
    ax.set_xlabel('Number of epochs', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_yscale('log')
    ax.set_xlim(0, 30000)
    ax.set_ylim(1.0e-6, 1.0e4)
    ax.set_xticks(np.arange(0, 35000, 10000))
    ax.tick_params(labelsize=10)
    
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    
    ax.legend(fontsize=10, ncol=2, loc='upper center', frameon=False)
    
    fig.tight_layout()
    fig.savefig('log/training_process_total_loss.pdf', dpi=300)
    
    return fig, ax

def plot_load_disp_fem_mlp_graph() -> Tuple[plt.Figure, plt.Axes]:
    result_folder = 'log/'
    fem_data = -np.loadtxt(result_folder + 'fem/load_disp.csv', delimiter=',')
    mlp_data = -np.loadtxt(result_folder + 'pinn/4x5/load_disp.csv', delimiter=',')[:26]
    graph_data = -np.loadtxt(result_folder + 'graph/3x2/load_disp.csv', delimiter=',')
    # exp_data = np.loadtxt('data/Experimental_data.csv', delimiter=',', skiprows=2)
    # telichko_data = np.loadtxt('data/Telichko.csv', delimiter=',', skiprows=2)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Plot FEM - black solid
    ax.plot(fem_data[:, 0], fem_data[:, 1], label='FEM', color=(255/255, 204/255, 0), linestyle='-',
            marker='o', linewidth=2.5, markersize=6, markerfacecolor='white', markeredgewidth=2)
    
    # Plot MLP - blue dashed with transparency
    area = 40*40
    load = (mlp_data[:, 1] * area) / 1000.0
    ax.plot(mlp_data[:, 0], load, label='MLP', color=(0, 47/255, 167/255), linestyle='-',
            marker='s', linewidth=2.5, markersize=4, markerfacecolor='white', markeredgewidth=2)
    rmse_mlp_disp = np.sqrt(np.mean((mlp_data[:, 0] - fem_data[:, 0]) ** 2))
    rmse_mlp_load = np.sqrt(np.mean((load - fem_data[:, 1]) ** 2))
    
    # Plot GNN - red dashed with transparency
    load = (graph_data[:, 1] * area) / 1000.0
    ax.plot(graph_data[:, 0], load, label='GNN', color=(192/255, 0, 0), linestyle='-',
            marker='D', linewidth=2.5, markersize=4, markerfacecolor='white', markeredgewidth=2)
    rmse_graph_disp = np.sqrt(np.mean((graph_data[:, 0] - fem_data[:, 0]) ** 2))
    rmse_graph_load = np.sqrt(np.mean((load - fem_data[:, 1]) ** 2))
    
    # Axis settings
    ax.set_xlabel('Displacement (mm)', fontsize=11)
    ax.set_ylabel('Load (kN)', fontsize=11)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(0, 0.4, 0.1))
    ax.set_yticks(np.arange(0, 70, 20))
    ax.tick_params(labelsize=10)
    
    # Customize tick direction
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    
    # Legend
    ax.legend(fontsize=10, loc='upper left', frameon=False)
    
    # print(rmse_mlp_load, rmse_graph_load)
    ax.text(0.15, 13,
        f"RMSE (MLP vs FEM):\n"
        f"   Displacement: {rmse_mlp_disp:.3f}, Load: {rmse_mlp_load:.1f}\n"
        f"RMSE (GNN vs FEM):\n"
        f"   Displacement: {rmse_graph_disp:.3f}, Load: {rmse_graph_load:.1f}",
        fontsize=10,
        color="black",
        ha="left",
        va="center",
        linespacing=1.5
        )
    
    # Save figure
    fig.tight_layout()
    fig.savefig(result_folder + 'response_comparison.pdf', dpi=300)
    
    return fig, ax

def plot_feature_importance() -> None:
    folder = 'log/graph/feature_importance/'
    pos = 'top'
    fi = np.loadtxt(folder + f'feature_importance_nodes_{pos}.csv', delimiter=',')
    F = fi.shape[1]

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([fi[:, j] for j in range(F)], showmeans=True, patch_artist=False)
    for box in bp['boxes']:
        box.set(color='blue', linewidth=1.5)

    # Axis labels
    ax.set_xlabel('Graph node feature (Coordinates)', fontsize=11)
    ax.set_ylabel('Normalized importance', fontsize=11)
    ax.set_ylim(0.25, 0.45)
    ax.set_yticks(np.arange(0.25, 0.5, 0.05))
    
    ax.set_xlim(0.5, 4)
    ax.set_xticks([i+1 for i in range(F)])
    ax.set_xticklabels([r"$x$", r"$y$", r"$z$"], fontsize=12)

    # Major ticks: keep both axes
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    # Minor ticks: only on y-axis
    ax.tick_params(axis='y', which='minor', direction='in', length=3, labelsize=0)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.xaxis.set_minor_locator(ticker.NullLocator())  # disables x-axis minor ticks

    # === Add annotations ===
    for j in range(F):
        # Get boxplot element coordinates
        ydata = fi[:, j]
        mean_val = np.mean(ydata)
        q1, q3 = np.percentile(ydata, [25, 75])

        # Annotate mean
        ax.annotate(f"μ={mean_val:.3f}", 
                    xy=(j+1.2, mean_val), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="green")

        # Annotate lower/upper box edges
        ax.annotate(f"Q1={q1:.3f}", 
                    xy=(j+1.1, q1-0.01), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="blue")

        ax.annotate(f"Q3={q3:.3f}", 
                    xy=(j+1.1, q3+0.009), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="blue")

    fig.tight_layout()
    fig.savefig(folder + f'fi_box_plot_{pos}.png', dpi=300, transparent=True)
    plt.close(fig)
    
    pos = 'inside'
    fi = np.loadtxt(folder + f'feature_importance_nodes_{pos}.csv', delimiter=',')
    F = fi.shape[1]

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([fi[:, j] for j in range(F)], showmeans=True, patch_artist=False)
    for box in bp['boxes']:
        box.set(color='blue', linewidth=1.5)

    # Axis labels
    ax.set_xlabel('Graph node feature (Coordinates)', fontsize=11)
    ax.set_ylabel('Normalized importance', fontsize=11)
    ax.set_ylim(0.25, 0.45)
    ax.set_yticks(np.arange(0.25, 0.5, 0.05))
    
    ax.set_xlim(0.5, 4)
    ax.set_xticks([i+1 for i in range(F)])
    ax.set_xticklabels([r"$x$", r"$y$", r"$z$"], fontsize=12)

    # Major ticks: keep both axes
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    # Minor ticks: only on y-axis
    ax.tick_params(axis='y', which='minor', direction='in', length=3, labelsize=0)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.xaxis.set_minor_locator(ticker.NullLocator())  # disables x-axis minor ticks

    # === Add annotations ===
    for j in range(F):
        # Get boxplot element coordinates
        ydata = fi[:, j]
        mean_val = np.mean(ydata)
        q1, q3 = np.percentile(ydata, [25, 75])

        # Annotate mean
        ax.annotate(f"μ={mean_val:.3f}", 
                    xy=(j+1.2, mean_val), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="green")

        # Annotate lower/upper box edges
        ax.annotate(f"Q1={q1:.3f}", 
                    xy=(j+1.1, q1-0.007), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="blue")

        ax.annotate(f"Q3={q3:.3f}", 
                    xy=(j+1.1, q3+0.005), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="blue")

    fig.tight_layout()
    fig.savefig(folder + f'fi_box_plot_{pos}.png', dpi=300, transparent=True)
    plt.close(fig)
    
    pos = 'bottom'
    fi = np.loadtxt(folder + f'feature_importance_nodes_{pos}.csv', delimiter=',')
    F = fi.shape[1]

    # Create two subplots, stacked vertically, sharing x-axis
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(5, 4),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # Plot boxplots on both axes
    data = [fi[:, j] for j in range(F)]
    bp_top = ax_top.boxplot(data, showmeans=True)
    bp_bottom = ax_bottom.boxplot(data, showmeans=True)
    for box in bp_top['boxes']:
        box.set(color='blue', linewidth=1.5)
    for box in bp_bottom['boxes']:
        box.set(color='blue', linewidth=1.5)

    # Set limits: keep top part and bottom part, cut out middle
    ax_top.set_ylim(0.45, 0.55)    # upper range
    ax_top.set_yticks(np.arange(0.45, 0.60, 0.05))
    ax_bottom.set_ylim(-0.05, 0.05) # lower range
    ax_bottom.set_yticks(np.arange(-0.05, 0.06, 0.05))

    # Hide the spines between the two plots
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False)  # don't put tick labels at the top subplot
    ax_bottom.xaxis.tick_bottom()

    # Diagonal slashes to indicate a break
    d = .005  # size of diagonal lines
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax_bottom.transAxes)  # switch to bottom axes
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels
    ax_bottom.set_xlabel("Graph node feature (Coordinates)", fontsize=11)
    # ax_top.set_ylabel("Normalized importance")
    # ax_bottom.set_ylabel("Normalized importance")
    fig.supylabel("Normalized importance", fontsize=11, x=0.045)
        
    ax_top.set_xlim(0.5, 4)
    ax_bottom.set_xlim(0.5, 4)
    ax_bottom.set_xticks([i+1 for i in range(F)])
    ax_bottom.set_xticklabels([r"$x$", r"$y$", r"$z$"], fontsize=12)
    
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    
    # Major ticks: keep both axes
    ax_bottom.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    # Minor ticks: only on y-axis
    ax_bottom.tick_params(axis='y', which='minor', direction='in', length=3, labelsize=0)
    ax_bottom.minorticks_on()
    ax_bottom.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax_bottom.xaxis.set_minor_locator(ticker.NullLocator())  # disables x-axis minor ticks
    ax_top.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    # Minor ticks: only on y-axis
    ax_top.tick_params(axis='y', which='minor', direction='in', length=3, labelsize=0)
    ax_top.minorticks_on()
    ax_top.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax_top.xaxis.set_minor_locator(ticker.NullLocator())  # disables x-axis minor ticks

    # === Add annotations ===
    for j in range(2):
        # Get boxplot element coordinates
        ydata = fi[:, j]
        mean_val = np.mean(ydata)
        q1, q3 = np.percentile(ydata, [25, 75])
        
        # Annotate mean
        ax_top.annotate(f"μ={mean_val:.3f}", 
                    xy=(j+1.2, mean_val), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="green")
        
        # Annotate lower/upper box edges
        ax_top.annotate(f"Q1={q1:.3f}", 
                    xy=(j+1.1, q1-0.007), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="blue")

        ax_top.annotate(f"Q3={q3:.3f}", 
                    xy=(j+1.1, q3+0.007), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="blue")
    
    j = 2
    ydata = fi[:, j]
    mean_val = np.mean(ydata)
    q1, q3 = np.percentile(ydata, [25, 75])
    
    # Annotate mean
    ax_bottom.annotate(f"μ={mean_val:.1f}", 
                xy=(j+1.2, mean_val), xycoords="data",
                xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color="green")
    
    # Annotate lower/upper box edges
    ax_bottom.annotate(f"Q1={q1:.1f}", 
                xy=(j+1.05, q1-0.02), xycoords="data",
                xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color="blue")

    ax_bottom.annotate(f"Q3={q3:.1f}", 
                xy=(j+1.05, q3+0.02), xycoords="data",
                xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color="blue")

    fig.tight_layout()
    fig.savefig(folder + f'fi_box_plot_{pos}.png', dpi=300, transparent=True)
    plt.close(fig)
    
    pos = 'left'
    fi = np.loadtxt(folder + f'feature_importance_nodes_{pos}.csv', delimiter=',')
    F = fi.shape[1]

    # Create two subplots, stacked vertically, sharing x-axis
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(5, 4),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # Plot boxplots on both axes
    data = [fi[:, j] for j in range(F)]
    bp_top = ax_top.boxplot(data, showmeans=True)
    bp_bottom = ax_bottom.boxplot(data, showmeans=True)
    for box in bp_top['boxes']:
        box.set(color='blue', linewidth=1.5)
    for box in bp_bottom['boxes']:
        box.set(color='blue', linewidth=1.5)

    # Set limits: keep top part and bottom part, cut out middle
    ax_top.set_ylim(0.4, 0.6)    # upper range
    ax_top.set_yticks(np.arange(0.4, 0.65, 0.1))
    ax_bottom.set_ylim(-0.05, 0.05) # lower range
    ax_bottom.set_yticks(np.arange(-0.05, 0.06, 0.05))

    # Hide the spines between the two plots
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False)  # don't put tick labels at the top subplot
    ax_bottom.xaxis.tick_bottom()

    # Diagonal slashes to indicate a break
    d = .005  # size of diagonal lines
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax_bottom.transAxes)  # switch to bottom axes
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels
    ax_bottom.set_xlabel("Graph node feature (Coordinates)", fontsize=11)
    # ax_top.set_ylabel("Normalized importance")
    # ax_bottom.set_ylabel("Normalized importance")
    fig.supylabel("Normalized importance", fontsize=11, x=0.045)
        
    ax_top.set_xlim(0.5, 4)
    ax_bottom.set_xlim(0.5, 4)
    ax_bottom.set_xticks([i+1 for i in range(F)])
    ax_bottom.set_xticklabels([r"$x$", r"$y$", r"$z$"], fontsize=12)
    
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    
    # Major ticks: keep both axes
    ax_bottom.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    # Minor ticks: only on y-axis
    ax_bottom.tick_params(axis='y', which='minor', direction='in', length=3, labelsize=0)
    ax_bottom.minorticks_on()
    ax_bottom.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax_bottom.xaxis.set_minor_locator(ticker.NullLocator())  # disables x-axis minor ticks
    ax_top.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    # Minor ticks: only on y-axis
    ax_top.tick_params(axis='y', which='minor', direction='in', length=3, labelsize=0)
    ax_top.minorticks_on()
    ax_top.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax_top.xaxis.set_minor_locator(ticker.NullLocator())  # disables x-axis minor ticks

    # === Add annotations ===
    for j in [0, 2]:
        # Get boxplot element coordinates
        ydata = fi[:, j]
        mean_val = np.mean(ydata)
        q1, q3 = np.percentile(ydata, [25, 75])
        
        # Annotate mean
        ax_top.annotate(f"μ={mean_val:.3f}", 
                    xy=(j+1.2, mean_val), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="green")
        
        # Annotate lower/upper box edges
        ax_top.annotate(f"Q1={q1:.3f}", 
                    xy=(j+1.1, q1-0.01), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="blue")

        ax_top.annotate(f"Q3={q3:.3f}", 
                    xy=(j+1.1, q3+0.01), xycoords="data",
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="blue")
    
    j = 1
    ydata = fi[:, j]
    mean_val = np.mean(ydata)
    q1, q3 = np.percentile(ydata, [25, 75])
    
    # Annotate mean
    ax_bottom.annotate(f"μ={mean_val:.1f}", 
                xy=(j+1.2, mean_val), xycoords="data",
                xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color="green")
    
    # Annotate lower/upper box edges
    ax_bottom.annotate(f"Q1={q1:.1f}", 
                xy=(j+1.05, q1-0.02), xycoords="data",
                xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color="blue")

    ax_bottom.annotate(f"Q3={q3:.1f}", 
                xy=(j+1.05, q3+0.02), xycoords="data",
                xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color="blue")

    fig.tight_layout()
    fig.savefig(folder + f'fi_box_plot_{pos}.png', dpi=300, transparent=True)
    plt.close(fig)
        
        

if __name__ == '__main__':
#     plot_load_disp_curves_step_load()
    # plot_training_process_mlp_pde()
    # plot_training_process_mlp_bcs()
    # plot_training_process_graph_pde()
    # plot_training_process_graph_bcs()
    # plot_training_process_total_loss()
    # plot_load_disp_fem_mlp_graph()
#     plot_feature_importance()
    pass
