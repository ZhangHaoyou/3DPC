# pc_fem/utils/utils.py

import numpy as np
import matplotlib
import platform

if platform.system() != 'Windows':
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import List, Union, Optional, Tuple


def local_to_global_index(node_tags: Union[List[int], np.ndarray], dof: int = 3) -> List[int]:
    """Computes the global index mapping for a given set of local node IDs.

    This function converts local node IDs into global indices used in 
    finite element global matrices by applying degree-of-freedom (DOF) mapping.

    Example:
        If `node_tags = [0, 1]` and `dof = 3`, the output will be:
        `[0, 1, 2, 3, 4, 5]` (Each node contributes DOF indices in order).

    Args:
        node_tags (List[int] or np.ndarray): List or array of node IDs (starting from 0).
        dof (int, optional): Degrees of freedom per node. Defaults to 3.

    Returns:
        List[int]: Global index list corresponding to the input node IDs.

    Raises:
        ValueError: If `node_tags` is empty or contains non-integer values.
    """

    # Validate input
    if not isinstance(node_tags, (list, np.ndarray)):
        raise ValueError("node_tags must be a list or numpy array of integers.")

    if len(node_tags) == 0:
        raise ValueError("node_tags cannot be empty.")

    if not all(isinstance(tag, (int, np.integer)) for tag in node_tags):
        raise ValueError("All elements in node_tags must be integers.")

    # Compute global indices using list comprehension
    global_indices = [int(tag * dof + i) for tag in node_tags for i in range(dof)]

    return global_indices

def calculate_principal_values(vec: np.ndarray, quantity: str = "strain") -> np.ndarray:
    """Computes the principal stresses or strains from a 6-component Voigt vector.

    This function converts the input Voigt vector into a symmetric 3×3 tensor 
    and calculates its principal values (eigenvalues), sorted in descending order.

    Args:
        vec (np.ndarray): A 1D array of shape (6,) representing stress or strain 
            in Voigt notation: [xx, yy, zz, xy, yz, xz].
        quantity (str): Type of input vector, either "strain" or "stress".
            For strain, the shear components are halved to obtain the correct tensor.
            Defaults to "strain".

    Returns:
        np.ndarray: A 1D array of shape (3,) containing the principal values (sorted descending).

    Raises:
        ValueError: If input is not a 6-component vector or if `quantity` is invalid.
    """
    if vec.shape != (6,):
        raise ValueError("Input vector must have shape (6,)")

    if quantity not in {"strain", "stress"}:
        raise ValueError("`quantity` must be either 'strain' or 'stress'")

    # Convert Voigt vector to tensor
    tensor = voigt_to_tensor(vec, quantity=quantity)

    # Use eigh for symmetric tensors (guaranteed real eigenvalues)
    eigenvalues, _ = np.linalg.eigh(tensor)

    return np.sort(eigenvalues)[::-1]

def macaulay_bracket(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Computes the Macaulay bracket of a scalar or array.

    The Macaulay bracket is defined as:
        ⟨x⟩ = max(x, 0)

    Commonly used in solid mechanics for tension-compression separation.

    Args:
        x (Union[float, np.ndarray]): Input scalar or array.

    Returns:
        Union[float, np.ndarray]: The Macaulay bracket of the input, with the same shape/type.
    """
    return np.maximum(x, 0.0)

def voigt_to_tensor(vec: np.ndarray, quantity: str = "strain") -> np.ndarray:
    """Converts a 6-component Voigt vector into a 3×3 symmetric tensor.

    This function transforms engineering strain or stress vectors from Voigt notation:
    [xx, yy, zz, xy, yz, xz]ᵀ into a 3×3 symmetric tensor. The shear terms are treated
    differently for strain and stress:
        - For strain: γ_ij is halved to obtain ε_ij.
        - For stress: shear components are used as-is.

    Args:
        vec (np.ndarray): A 1D array of shape (6,) in Voigt notation: 
            [xx, yy, zz, xy, yz, xz].
        quantity (str): Type of input vector, either "strain" or "stress".
            If "strain", shear components are halved. Defaults to "strain".

    Returns:
        np.ndarray: A 3×3 symmetric tensor.

    Raises:
        ValueError: If `vec` is not of shape (6,) or if `quantity` is invalid.
    """
    if vec.shape != (6,):
        raise ValueError("Input vector must have shape (6,)")

    if quantity.lower() not in {"strain", "stress"}:
        raise ValueError("`quantity` must be either 'strain' or 'stress'")

    sxy, syz, sxz = vec[3], vec[4], vec[5]

    if quantity == "strain":
        sxy *= 0.5
        syz *= 0.5
        sxz *= 0.5

    tensor = np.array([
        [vec[0], sxy, sxz],
        [sxy,    vec[1], syz],
        [sxz,    syz,    vec[2]]
    ])

    return tensor

def calculate_equivalent_strain_from_voigt(vec: np.ndarray) -> float:
    """Computes the Mazars-type equivalent strain from a 6-component Voigt strain vector.

    The function converts the Voigt strain vector to a symmetric strain tensor, computes
    its principal strains, applies the Macaulay bracket (⟨·⟩ = max(x, 0)) to isolate
    tensile components, and returns the scalar equivalent strain.

    This is commonly used in isotropic damage models like the Mazars model.

    Args:
        vec (np.ndarray): A 1D array of shape (6,) representing the engineering strain
            in Voigt notation: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz].

    Returns:
        float: The scalar equivalent strain, considering only tensile principal strains.

    Raises:
        ValueError: If input vector does not have shape (6,).
    """
    if vec.shape != (6,):
        raise ValueError("Input strain vector must have shape (6,)")

    # Convert to strain tensor (apply 0.5 to shear components)
    eps_tensor = voigt_to_tensor(vec, quantity="strain")

    # Compute principal strains (eigenvalues)
    eps_principal = np.linalg.eigvalsh(eps_tensor)

    # Apply Macaulay bracket to get tensile components
    eps_pos = macaulay_bracket(eps_principal)

    # Compute equivalent strain
    eq_strain = np.sqrt(np.sum(eps_pos**2))
    
    return float(eq_strain)

# Visualization
def save_to_vtu(filename: str, nodes: np.ndarray, elements: np.ndarray,
                scalar: Optional[np.ndarray] = None, scalar_name: str = 'stress') -> None:
    """Saves a hexahedral mesh and associated scalar data to a VTU file for visualization in ParaView.

    This function exports a finite element mesh with nodal or elemental data to the VTU format.
    - **Geometric data** (coordinates, displacements) use shared nodes across elements.
    - **Physical data** (stress, strain) use independent nodes for each element, preventing averaging.

    Args:
        filename (str): Name of the output VTU file.
        nodes (np.ndarray): Nodal coordinates, shape (N, 3).
        elements (np.ndarray): Element connectivity (node indices per element), shape (M, 8).
        scalar (np.ndarray, optional): Elemental or nodal scalar values for visualization. Defaults to None.
        scalar_name (str, optional): Name of the scalar field. Defaults to 'stress'.

    Returns:
        None: The function saves the VTU file and does not return a value.

    Raises:
        ValueError: If `nodes` shape is not (N, 3).
        ValueError: If `elements` shape is not (M, 8).
        ValueError: If `scalar` is provided but does not match the expected size.
    """
    import pyvista as pv
    # Validate input shapes
    if nodes.shape[1] != 3:
        raise ValueError(f"Nodes must have shape (N, 3), but got {nodes.shape}.")
    
    if elements.shape[1] != 8:
        raise ValueError(f"Elements must have shape (M, 8), but got {elements.shape}.")
    
    # Check if scalar data is provided and has the correct shape
    if scalar is not None:
        expected_size = len(elements) * 8  # Since each element has 8 nodes
        scalar = scalar.ravel()
        if scalar.shape[0] not in [len(nodes), expected_size]:
            raise ValueError(f"Scalar data size must be {len(nodes)} (nodal) or {expected_size} (elemental), but got {scalar.shape}.")
    
    M = len(elements)  # Number of elements
    # Case 1: Geometric Data (e.g., displacement, coordinates) -> Shared Nodes
    if 'disp' in scalar_name.lower() or 'geo' in scalar_name.lower():
        # VTU format requires explicit cell connectivity storage
        cell_array = np.hstack([np.full((M, 1), 8, dtype=int), elements])
        cell_types = np.full(M, pv.CellType.HEXAHEDRON, dtype=np.uint8)

        # Create unstructured grid
        grid = pv.UnstructuredGrid(cell_array, cell_types, nodes)

        # Attach scalar data if available
        if scalar is not None:
            grid.point_data[scalar_name] = scalar  # Stored at shared nodes
    else:
        # Case 2: Physical Data (e.g., stress, strain) -> Independent Nodes
        

        # 1. Expand nodes so that each element has independent nodes
        expanded_nodes = nodes[elements.flatten()]  # Flatten elements to obtain corresponding node coordinates
        expanded_elements = np.arange(len(expanded_nodes)).reshape(M, 8)  # Re-index new nodes per element

        # 2. Construct VTU cell format
        cell_array = np.hstack([np.full((M, 1), 8, dtype=int), expanded_elements])
        cell_types = np.full(M, pv.CellType.HEXAHEDRON, dtype=np.uint8)

        # 3. Create an UnstructuredGrid object
        grid = pv.UnstructuredGrid(cell_array, cell_types, expanded_nodes)

        # 4. Store stress/strain as point data (each element has independent nodes)
        if scalar is not None:
            grid.point_data[scalar_name] = scalar  # No averaging occurs in ParaView

    # Save the grid to a VTU file
    grid.save(filename)

def plot_template(*args: List[Tuple[np.ndarray, np.ndarray]], 
            figsize: Tuple[float, float] = (8/2.54, 5.5/2.54),
            legend_labels: Optional[List[str]] = None,
            xlabel: str = 'X-Axis', ylabel: str = 'Y-Axis',
            colors: Optional[List[str]] = None, 
            linestyles: Optional[List[str]] = None, 
            markers: Optional[List[str]] = None, 
            markeverys: Optional[List[int]] = None,
            xlim: Optional[Tuple[float, float]] = None,
            major_x_interval: Optional[float] = None,
            ylim: Optional[Tuple[float, float]] = None,
            major_y_interval: Optional[float] = None,
            title: str = '',
            legend_in_row: bool = False,
            xtick_precision: Optional[int] = None, 
            ytick_precision: Optional[int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Generates a customizable line or scatter plot using Matplotlib.

    This function provides a flexible template for generating line plots, scatter plots,
    or combined point-line plots with customizable markers, colors, axis limits, tick intervals, and legends.

    Args:
        *args: Variable-length argument list for input data, formatted as [[x1, y1], [x2, y2], ...].
        figsize (Tuple[float, float], optional): Figure size in cm (default: (8, 5)).
        legend_labels (List[str], optional): List of labels for the legend.
        xlabel (str, optional): Label for the x-axis (default: 'X-Axis').
        ylabel (str, optional): Label for the y-axis (default: 'Y-Axis').
        colors (List[str], optional): List of colors for each data series.
        linestyles (List[str], optional): List of line styles for each data series.
        markers (List[str], optional): List of markers for each data series.
        markeverys (List[int], optional): List of interval values for marker placement.
        xlim (Tuple[float, float], optional): X-axis limits (default: None).
        major_x_interval (float, optional): Interval for major x-ticks.
        ylim (Tuple[float, float], optional): Y-axis limits (default: None).
        major_y_interval (float, optional): Interval for major y-ticks.
        title (str, optional): Title of the plot (default: '').
        legend_in_row (bool, optional): If True, displays the legend in a single row (default: False).
        xtick_precision (int, optional): Decimal precision for x-tick labels.
        ytick_precision (int, optional): Decimal precision for y-tick labels.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The generated Matplotlib figure and axes.

    Raises:
        ValueError: If `args` is empty or improperly formatted.
        ValueError: If `xlim` or `ylim` is provided but not as a tuple of two values.
    """
    
    # Validate input data
    if not args:
        raise ValueError("At least one dataset (x, y) must be provided in `args`.")
    
    for data in args:
        if not isinstance(data, (list, tuple)) or len(data) != 2:
            raise ValueError("Each dataset in `args` must be a tuple or list containing two arrays: (x, y).")
    
    # Validate axis limits
    if xlim is not None and (not isinstance(xlim, tuple) or len(xlim) != 2):
        raise ValueError("`xlim` must be a tuple containing two values (min, max).")
    if ylim is not None and (not isinstance(ylim, tuple) or len(ylim) != 2):
        raise ValueError("`ylim` must be a tuple containing two values (min, max).")
    
    # Set Matplotlib font styles
    if platform.system() == 'Windows':
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'mathtext.fontset': 'custom',
            'mathtext.rm': 'Times New Roman',
            'mathtext.it': 'Times New Roman:italic',
            'mathtext.bf': 'Times New Roman:bold',
        })
    
    # Convert figure size from cm to inches (1 cm = 1/2.54 inches)
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]), dpi=200)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=11)
    
    # Plot each dataset
    for i, (x, y) in enumerate(args):
        ax.plot(
            x, y, 
            label=legend_labels[i] if legend_labels else None,
            color=colors[i] if colors else None,
            linestyle=linestyles[i] if linestyles else None,
            marker=markers[i] if markers else None,
            markevery=markeverys[i] if markeverys else None,
            markersize=4, markeredgewidth=1, markeredgecolor=colors[i] if colors else None,
            markerfacecolor='white', zorder=3
        )
    
    # Set axis labels
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    # Set custom axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Set major and minor tick intervals
    if major_x_interval and xlim:
        xticks = np.arange(xlim[0], xlim[1] + major_x_interval, major_x_interval)
        if xtick_precision:
            xticks = np.round(xticks, xtick_precision)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in xticks], fontsize=8)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    if major_y_interval and ylim:
        yticks = np.arange(ylim[0], ylim[1] + major_y_interval, major_y_interval)
        if ytick_precision:
            yticks = np.round(yticks, ytick_precision)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(i) for i in yticks], fontsize=8)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    # Customize tick direction
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))

    # Add legend
    if legend_labels:
        ax.legend(fontsize=10, frameon=False, ncol=len(legend_labels) if legend_in_row else 1,
                handletextpad=0.2, markerscale=0.8, columnspacing=0.5)

    plt.tight_layout()
    
    return fig, ax

def annotate_point(
        x: float,
        y: float,
        fig: plt.Figure,
        ax: plt.Axes,
        marker_size: int = 7,
        marker_color: Union[str, Tuple[float, float, float]] = (192/255, 0, 0),
        text_size: int = 8,
        text_color: Union[str, Tuple[float, float, float]] = (0, 47/255, 167/255),
        ha: str = "left",
        va: str = "top",
        decimal_places_x: int = 2,
        decimal_places_y: int = 2
        ) -> Tuple[plt.Figure, plt.Axes]:
    """Annotates a point (x, y) in a Matplotlib figure with a marker and a text label.

    Args:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        fig (plt.Figure): Matplotlib figure object.
        ax (plt.Axes): Matplotlib axes object.
        marker_size (int, optional): Size of the marker. Defaults to 7.
        marker_color (Union[str, Tuple[float, float, float]], optional): 
            Color of the marker, either a string (e.g., 'red') or an RGB tuple (e.g., (0.75, 0, 0)).
            Defaults to (192/255, 0, 0) (a dark red).
        text_size (int, optional): Font size of the annotation text. Defaults to 8.
        text_color (Union[str, Tuple[float, float, float]], optional): 
            Color of the text, either a string (e.g., 'blue') or an RGB tuple (e.g., (0, 0.18, 0.65)).
            Defaults to (0, 47/255, 167/255) (a dark blue).
        ha (str, optional): Horizontal alignment of the text. Defaults to "left".
        va (str, optional): Vertical alignment of the text. Defaults to "top".
        decimal_places_x (int, optional): Number of decimal places to display for the X-coordinate. Defaults to 2.
        decimal_places_y (int, optional): Number of decimal places to display for the Y-coordinate. Defaults to 2.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The generated Matplotlib figure and axes.

    Raises:
        ValueError: If `x` or `y` is not a numeric value.
        ValueError: If `decimal_places` is negative.
    """
    # Validate x and y as numeric values
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Both `x` and `y` must be numeric values.")

    # Validate decimal_places_x and decimal_places_y
    if not isinstance(decimal_places_x, int) or decimal_places_x < 0:
        raise ValueError("`decimal_places_x` must be a non-negative integer.")
    if not isinstance(decimal_places_y, int) or decimal_places_y < 0:
        raise ValueError("`decimal_places_y` must be a non-negative integer.")

    # Format text with dynamic decimal places
    formatted_text = f" ({x:.{decimal_places_x}f}, {y:.{decimal_places_y}f})"

    # Plot the point
    ax.scatter([x], [y], color=marker_color, s=marker_size, zorder=3)

    # Add the text annotation
    ax.text(x, y, formatted_text, fontsize=text_size, color=text_color, ha=ha, va=va)
    
    return fig, ax