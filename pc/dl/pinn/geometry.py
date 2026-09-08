# dl/pinn/geometry.py


import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from abc import ABC, abstractmethod
from typing import Tuple, List


from fem.parameters import Geometry_Parameter


class PINN_Geometry(ABC):
    """Abstract base class for PINN problem geometry.
    
    Defines the interface for sampling collocation and boundary points,
    both as raw tensors and as graph representations, to be used in
    physics-informed neural network training.
    
    Attributes:
        device (str): Device tag for torch tensors (e.g. 'cpu' or 'cuda:0').
        dtype (torch.dtype): Data type for created tensors.
    """
    def __init__(self, device: str, dtype) -> None:
        """Initialize geometry with target device and data type.
        
        Args:
            device (str): Computation device for tensors.
            dtype (torch.dtype): Desired torch data type.
        """
        self.device = device
        self.dtype  = dtype
    
    def _generate_random_points_uniform(self, N: int, seed: int = 0, eps: float = 1e-4) -> torch.Tensor:
        """Generate N random normalized points in [eps, 1 - eps] along one axis.
        
        Seeds the torch RNG for reproducibility, then samples uniformly from
        the closed interval [eps, 1 - eps] so that no value lies exactly on 0 or 1.
        
        Args:
            N (int): Number of points to generate.
            seed (int): Random seed for reproducibility.
            eps (float): Small offset to avoid exact boundary values.
            
        Returns:
            torch.Tensor: 1D tensor of shape (N,), on `self.device` with
                dtype `self.dtype`, containing samples in [eps, 1 - eps].
        """
        # set seed for reproducible sampling
        torch.manual_seed(seed)
        # uniform sample in [eps, 1 - eps], ensure correct dtype & device
        return torch.rand(N, device=self.device) * (1 - 2 * eps) + eps
    
    def _generate_a_list_of_points(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N normalized sample points in (0, 1) using the specified method.
        
        This will produce a 1D tensor of length N on `self.device` with dtype `self.dtype`.
        
        Args:
            N (int): Number of points to generate.
            method (str): Sampling strategy to use. One of:
                - 'linspace': N equally spaced points in (0, 1).
                - 'uniform': N random points in [eps, 1-eps] via
                `_generate_random_points_uniform`; uses the helper’s default eps.
                - 'lhs': Latin-hypercube sampling in (0, 1).
            seed (int): RNG seed for reproducibility (used by 'uniform' and 'lhs').

        Returns:
            torch.Tensor: 1D tensor of shape (N,), dtype `self.dtype`, device `self.device`.
            
        Raises:
            ValueError: If `method` is not one of the supported options.
        """
        if method == 'linspace':
            # N+2 points from 0 -> 1, then trim endpoints -> N interior pts
            return torch.linspace(0.0, 1.0, N + 2, device=self.device, dtype=self.dtype)[1:-1]
        elif method == 'uniform':
            # sample in [eps, 1−eps]
            return self._generate_random_points_uniform(N=N, seed=seed)
        elif method == 'lhs':
            from pyDOE import lhs
            # seed numpy for reproducibility
            np.random.seed(seed)
            # lhs(1, samples=N) -> shape (N,1)
            arr = lhs(1, samples=N).ravel()
            return torch.as_tensor(arr, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Unknown sampling method '{method}'")
    
    @abstractmethod
    def generate_random_points_inside(N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N random points strictly inside the domain.
        
        Args:
            N (int): Number of points to generate.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N, D) containing interior point coordinates.
        """
        pass
    
    @abstractmethod
    def generate_random_points_on_top(N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N random points on the top boundary face.
        
        Args:
            N (int): Number of points to generate.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N, D) with top-face coordinates.
        """
        pass
    
    @abstractmethod
    def generate_random_points_on_bottom(N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N random points on the bottom boundary face.
        
        Args:
            N (int): Number of points to generate.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N, D) with bottom-face coordinates.
        """
        pass
    
    @abstractmethod
    def generate_random_points_on_front(N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N random points on the front boundary face.
        
        Args:
            N (int): Number of points to generate.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N, D) with front-face coordinates.
        """
        pass
    
    @abstractmethod
    def generate_random_points_on_back(N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N random points on the back boundary face.
        
        Args:
            N (int): Number of points to generate.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N, D) with back-face coordinates.
        """
        pass
    
    @abstractmethod
    def generate_random_points_on_right(N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N random points on the right boundary face.
        
        Args:
            N (int): Number of points to generate.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N, D) with right-face coordinates.
        """
        pass
    
    @abstractmethod
    def generate_random_points_on_left(N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N random points on the left boundary face.
        
        Args:
            N (int): Number of points to generate.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N, D) with left-face coordinates.
        """
        pass
    
    def generate_inside_graph(self, N: int, method: str, seed: int, k: int) -> Data:
        """Generate a k-NN graph over N interior points.
        
        Args:
            N (int): Number of interior points to sample.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            k (int): Number of nearest neighbors per node.
            
        Returns:
            Data: A torch_geometric Data object with:
                - x (Tensor): Node feature tensor of shape (N, D).
                - edge_index (LongTensor): Edge index of shape (2, E).
        """
        inside_points = self.generate_random_points_inside(N=N, method=method, seed=seed)
        edge_index = knn_graph(x=inside_points, k=k, loop=False)
        data_inside = Data(x=inside_points, edge_index=edge_index)
        return data_inside
    
    def generate_top_graph(self, N: int, method: str, seed: int, k: int) -> Data:
        """Generate a k-NN graph over N points on the top boundary face.
        
        Args:
            N (int): Number of boundary points to sample.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            k (int): Number of nearest neighbors per node.
            
        Returns:
            Data: A torch_geometric Data object with:
                - x (Tensor): Node feature tensor of shape (N, D).
                - edge_index (LongTensor): Edge index of shape (2, E).
        """
        top_points = self.generate_random_points_on_top(N=N, method=method, seed=seed)
        edge_index = knn_graph(x=top_points, k=k, loop=False)
        data_top = Data(x=top_points, edge_index=edge_index)
        return data_top
    
    def generate_bottom_graph(self, N: int, method: str, seed: int, k: int) -> Data:
        """Generate a k-NN graph over N points on the bottom boundary face.
        
        Args:
            N (int): Number of boundary points to sample.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            k (int): Number of nearest neighbors per node.
            
        Returns:
            Data: A torch_geometric Data object with:
                - x (Tensor): Node feature tensor of shape (N, D).
                - edge_index (LongTensor): Edge index of shape (2, E).
        """
        bottom_points = self.generate_random_points_on_bottom(N=N, method=method, seed=seed)
        edge_index = knn_graph(x=bottom_points, k=k, loop=False)
        data_bottom = Data(x=bottom_points, edge_index=edge_index)
        return data_bottom
    
    def generate_front_graph(self, N: int, method: str, seed: int, k: int) -> Data:
        """Generate a k-NN graph over N points on the front boundary face.
        
        Args:
            N (int): Number of boundary points to sample.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            k (int): Number of nearest neighbors per node.
            
        Returns:
            Data: A torch_geometric Data object with:
                - x (Tensor): Node feature tensor of shape (N, D).
                - edge_index (LongTensor): Edge index of shape (2, E).
        """
        front_points = self.generate_random_points_on_front(N=N, method=method, seed=seed)
        edge_index = knn_graph(x=front_points, k=k, loop=False)
        data_front = Data(x=front_points, edge_index=edge_index)
        return data_front
    
    def generate_back_graph(self, N: int, method: str, seed: int, k: int) -> Data:
        """Generate a k-NN graph over N points on the back boundary face.
        
        Args:
            N (int): Number of boundary points to sample.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            k (int): Number of nearest neighbors per node.
            
        Returns:
            Data: A torch_geometric Data object with:
                - x (Tensor): Node feature tensor of shape (N, D).
                - edge_index (LongTensor): Edge index of shape (2, E).
        """
        back_points = self.generate_random_points_on_back(N=N, method=method, seed=seed)
        edge_index = knn_graph(x=back_points, k=k, loop=False)
        data_back = Data(x=back_points, edge_index=edge_index)
        return data_back
    
    def generate_right_graph(self, N: int, method: str, seed: int, k: int) -> Data:
        """Generate a k-NN graph over N points on the right boundary face.
        
        Args:
            N (int): Number of boundary points to sample.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            k (int): Number of nearest neighbors per node.
            
        Returns:
            Data: A torch_geometric Data object with:
                - x (Tensor): Node feature tensor of shape (N, D).
                - edge_index (LongTensor): Edge index of shape (2, E).
        """
        right_points = self.generate_random_points_on_right(N=N, method=method, seed=seed)
        edge_index = knn_graph(x=right_points, k=k, loop=False)
        data_right = Data(x=right_points, edge_index=edge_index)
        return data_right
    
    def generate_left_graph(self, N: int, method: str, seed: int, k: int) -> Data:
        """Generate a k-NN graph over N points on the left boundary face.
        
        Args:
            N (int): Number of boundary points to sample.
            method (str): Sampling strategy name.
            seed (int): Random seed for reproducibility.
            k (int): Number of nearest neighbors per node.
            
        Returns:
            Data: A torch_geometric Data object with:
                - x (Tensor): Node feature tensor of shape (N, D).
                - edge_index (LongTensor): Edge index of shape (2, E).
        """
        left_points = self.generate_random_points_on_left(N=N, method=method, seed=seed)
        edge_index = knn_graph(x=left_points, k=k, loop=False)
        data_left = Data(x=left_points, edge_index=edge_index)
        return data_left
    
    def combine_graphs(self, graph_list: List[Data]) -> Data:
        """Merge multiple graphs into a single graph.
        
        Args:
            graph_list (List[Data]): List of torch_geometric Data objects to merge.
                Each Data.x holds node features, and each Data.edge_index holds edges
                indexed relative to its own nodes.
                
        Returns:
            Data: A combined Data object where:
                - x is the concatenation of all node feature tensors.
                - edge_index is the concatenation of all edge indices, offset so that
                    indices refer to the new, unified node set.
        """
        all_x = []
        all_edge_indices = []
        offset = 0
        for graph in graph_list:
            num_nodes = graph.x.size(0)
            all_x.append(graph.x)
            # Adjust edge_index by current node offset
            edge_index = graph.edge_index + offset
            all_edge_indices.append(edge_index)
            offset += num_nodes
        combined_x = torch.cat(all_x, dim=0)
        combined_edge_index = torch.cat(all_edge_indices, dim=1)  # concat along edges
        combined_graph = Data(x=combined_x, edge_index=combined_edge_index)
        return combined_graph

class Single_Cube_Geometry(PINN_Geometry):
    """Axis-aligned bounding-box geometry for a single contact region.

    This class represents a rectangular prism in 3D defined by its minimum
    and maximum corner coordinates. It provides methods to:
    
      * Test whether points lie inside or on any face.
      * Compute normals and tangents on the boundary.
      * Sample random points on faces or inside.
      * Build k-NN graphs over those samples.

    Attributes:
        bbox (torch.Tensor): Tensor of shape (2, 3), where
            row 0 = (xmin, ymin, zmin) and row 1 = (xmax, ymax, zmax).
        mins (torch.Tensor): 3-element tensor of minimum coords.
        maxs (torch.Tensor): 3-element tensor of maximum coords.
        device (torch.device): Device of the underlying tensors.
    """
    def __init__(self, bbox: torch.Tensor) -> None:
        """Initialize a new Single_Cube_Geometry.
        
        Args:
            bbox (torch.Tensor): Tensor of shape (2, 3) giving the
                minimum and maximum corners:
                  * bbox[0] = (xmin, ymin, zmin)
                  * bbox[1] = (xmax, ymax, zmax)
                Must already reside on the intended device.
        """
        if bbox.ndim != 2 or bbox.shape[1] != 3 or bbox.shape[0] != 2:
            raise ValueError(f"`bbox` must be shape (2,3), got {tuple(bbox.shape)}")
        # store full box and split into mins/maxs
        self.bbox = bbox
        self.mins = self.bbox[0]
        self.maxs = self.bbox[1]
        
        self.device = bbox.device
        super().__init__(device=bbox.device, dtype=bbox.dtype)
    
    def are_inside(self, x: torch.Tensor) -> torch.BoolTensor:
        """Test which points lie strictly inside the box.
        
        Args:
            x: Float tensor of shape (N,3) in physical coords.
            
        Returns:
            Bool tensor of shape (N,) where True indicates
            xmin < x < xmax on all coordinates.
        """
        gt_min = x > self.mins     # (N,3), broadcasts mins
        lt_max = x < self.maxs     # (N,3), broadcasts maxs
        return (gt_min & lt_max).all(dim=-1)
    
    def are_on_top(self, x: torch.Tensor, atol: float = 1e-6) -> torch.BoolTensor:
        """Return a mask of which points lie (within tolerance) on the top face z = z_max.
        
        Args:
            x: Float tensor of shape (N,3) in physical coords.
            atol: Absolute tolerance when comparing z-values.
            
        Returns:
            Bool tensor of shape (N,), True where |x[i,2] - z_max| ≤ atol.
        """
        z_top = self.maxs[2]
        return torch.isclose(x[:, 2], z_top, atol=atol)
        
    def are_on_bottom(self, x: torch.Tensor, atol: float = 1e-6) -> torch.BoolTensor:
        """Return a mask of which points lie (within tolerance) on the bottom face z = z_min.
        
        Args:
            x: Float tensor of shape (N,3) in physical coordinates.
            atol: Absolute tolerance for the z-comparison.
            
        Returns:
            Bool tensor of shape (N,) where True indicates
            |x[i,2] - z_min| ≤ atol.
        """
        z_min = self.mins[2]
        return torch.isclose(x[:, 2], z_min, atol=atol)
    
    def are_on_front(self, x: torch.Tensor, atol: float = 1e-6) -> torch.BoolTensor:
        """Return a mask of which points lie (within tolerance) on the front face x = x_max.
        
        Args:
            x: Float tensor of shape (N,3) in physical coordinates.
            atol: Absolute tolerance for the x-comparison.
            
        Returns:
            Bool tensor of shape (N,) where True indicates
            |x[i,0] - x_max| ≤ atol.
        """
        x_front = self.maxs[0]
        return torch.isclose(x[:, 0], x_front, atol=atol)
    
    def are_on_back(self, x: torch.Tensor, atol: float = 1e-6) -> torch.BoolTensor:
        """Return a mask of which points lie (within tolerance) on the back face x = x_min.
        
        Args:
            x: Float tensor of shape (N,3) in physical coordinates.
            atol: Absolute tolerance for the x-comparison.
            
        Returns:
            Bool tensor of shape (N,) where True indicates
            |x[i,0] - x_min| ≤ atol.
        """
        x_back = self.mins[0]
        return torch.isclose(x[:, 0], x_back, atol=atol)
    
    def are_on_right(self, x: torch.Tensor, atol: float = 1e-6) -> torch.BoolTensor:
        """Return a mask of which points lie (within tolerance) on the right face y = y_max.
        
        Args:
            x: Float tensor of shape (N,3) in physical coordinates.
            atol: Absolute tolerance for the y-comparison.
            
        Returns:
            Bool tensor of shape (N,) where True indicates
            |x[i,1] - y_max| ≤ atol.
        """
        y_right = self.maxs[1]
        return torch.isclose(x[:, 1], y_right, atol=atol)
    
    def are_on_left(self, x: torch.Tensor, atol: float = 1e-6) -> torch.BoolTensor:
        """Return a mask of which points lie (within tolerance) on the left face y = y_min.
        
        Args:
            x: Float tensor of shape (N,3) in physical coordinates.
            atol: Absolute tolerance for the y-comparison.
            
        Returns:
            Bool tensor of shape (N,) where True indicates
            |x[i,1] - y_min| ≤ atol.
        """
        y_left = self.mins[1]
        return torch.isclose(x[:, 1], y_left, atol=atol)
    
    def calculate_boundary_normals_tangents(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute outward normals and two orthogonal tangents on each box face.
        
        Given N points `x` on (or off) the boundary of this single-contact box,
        returns three (N,3) tensors `normals, t1, t2` such that for any point on
        a face, it gets that face's normal and two tangent directions; interior
        points remain zero.
        
        Args:
            x: FloatTensor of shape (N,3), physical coordinates.
            
        Returns:
            normals: (N,3) outward unit normals (zeros off the boundary).  
            t1:      (N,3) first tangent on that face.  
            t2:      (N,3) second tangent on that face.
        """
        # allocate once, on the same device/dtype as x
        N = x.shape[0]
        normals = x.new_zeros((N, 3))
        t1      = x.new_zeros((N, 3))
        t2      = x.new_zeros((N, 3))
        
        # pre-create each face’s (normal, tangent1, tangent2)
        n_top    = x.new_tensor([0.0,  0.0,  1.0])
        t1_top   = x.new_tensor([1.0,  0.0,  0.0])
        t2_top   = x.new_tensor([0.0,  1.0,  0.0])
        
        n_bottom = x.new_tensor([0.0,  0.0, -1.0])
        t1_bottom= x.new_tensor([0.0,  1.0,  0.0])
        t2_bottom= x.new_tensor([1.0,  0.0,  0.0])
        
        n_front  = x.new_tensor([1.0,  0.0,  0.0])
        t1_front = x.new_tensor([0.0,  1.0,  0.0])
        t2_front = x.new_tensor([0.0,  0.0,  1.0])
        
        n_back   = x.new_tensor([-1.0, 0.0,  0.0])
        t1_back  = x.new_tensor([0.0,  0.0,  1.0])
        t2_back  = x.new_tensor([0.0,  1.0,  0.0])
        
        n_right  = x.new_tensor([0.0,  1.0,  0.0])
        t1_right = x.new_tensor([0.0,  0.0,  1.0])
        t2_right = x.new_tensor([1.0,  0.0,  0.0])
        
        n_left   = x.new_tensor([0.0, -1.0,  0.0])
        t1_left  = x.new_tensor([1.0,  0.0,  0.0])
        t2_left  = x.new_tensor([0.0,  0.0,  1.0])
        
        # assign per-face
        m = self.are_on_top(x)
        normals[m] = n_top;    t1[m] = t1_top;    t2[m] = t2_top
        
        m = self.are_on_bottom(x)
        normals[m] = n_bottom; t1[m] = t1_bottom; t2[m] = t2_bottom
        
        m = self.are_on_front(x)
        normals[m] = n_front;  t1[m] = t1_front;  t2[m] = t2_front
        
        m = self.are_on_back(x)
        normals[m] = n_back;   t1[m] = t1_back;   t2[m] = t2_back
        
        m = self.are_on_right(x)
        normals[m] = n_right;  t1[m] = t1_right;  t2[m] = t2_right
        
        m = self.are_on_left(x)
        normals[m] = n_left;   t1[m] = t1_left;   t2[m] = t2_left
        
        return normals, t1, t2
    
    def generate_random_points_inside(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample N**3 points inside [self.mins, self.maxs] via per-axis sampling + Cartesian product.
        
        For 'linspace' and 'uniform', each axis uses
        `self._generate_a_list_of_points`, which yields N values in (0,1);
        for 'lhs', Latin-hypercube sampling in 3D produces N values per axis.
        Finally, normalized values are mapped into (mins, maxs).
        
        Args:
            N:      Number of samples per axis.
            method: {'linspace','uniform','lhs'} sampling strategy.
            seed:   RNG seed for reproducibility.
            
        Returns:
            Tensor of shape (N**3, 3), dtype=self.dtype,
            device=self.device, with points in the physical box.
        """
        # 1) get normalized samples in (0,1)
        if method in ('linspace', 'uniform'):
            x = self._generate_a_list_of_points(N=N, method=method, seed=seed)
            y = self._generate_a_list_of_points(N=N, method=method, seed=seed)
            z = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        elif method == 'lhs':
            from pyDOE import lhs
            np.random.seed(seed)
            samples = lhs(3, samples=N)  # shape (N,3), values in (0,1)
            x = torch.tensor(samples[:, 0], dtype=self.dtype, device=self.device)
            y = torch.tensor(samples[:, 1], dtype=self.dtype, device=self.device)
            z = torch.tensor(samples[:, 2], dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Unknown method '{method}'")
        # 2) scale normalized coords into physical box
        span = self.maxs - self.mins  # (3,)
        x_phys = x * span[0] + self.mins[0]
        y_phys = y * span[1] + self.mins[1]
        z_phys = z * span[2] + self.mins[2]
        # 3) form Cartesian product -> (N**3, 3)
        return torch.cartesian_prod(x_phys, y_phys, z_phys)
    
    def generate_random_points_on_top(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N**2 points on the top face (z = z_max) of the 3D box.
        
        1) Sample N values in (0,1) along x and y via your chosen method.
        2) Fix normalized z = 1.0 (top).
        3) Map (x_norm,y_norm,1) into physical coords (mins, maxs).
        4) Take the Cartesian product -> (N**2,3).
        
        Args:
            N: Number of points per horizontal axis.
            method: 'linspace', 'uniform', or 'lhs'.
            seed: RNG seed for reproducibility.
            
        Returns:
            Tensor of shape (N**2, 3), dtype=self.mins.dtype, device=self.device,
            with x∈(xmin,xmax), y∈(ymin,ymax), z=z_max.
        """
        # 1) get normalized 1D samples in (0,1)
        x_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        y_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        z_norm = torch.ones(1, device=self.device, dtype=self.mins.dtype)  # normalized top
        # 2) map into physical box
        span = self.maxs - self.mins                # (3,)
        x_phys = x_norm * span[0] + self.mins[0]    # (N,)
        y_phys = y_norm * span[1] + self.mins[1]    # (N,)
        z_phys = z_norm * span[2] + self.mins[2]    # (1,) -> equals self.maxs[2]
        # 3) Cartesian product -> (N**2, 3)
        return torch.cartesian_prod(x_phys, y_phys, z_phys)
    
    def generate_random_points_on_bottom(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N**2 points on the bottom face (z = z_min) of the 3D box.
        
        1) Sample N values in (0,1) along x and y via your chosen method.
        2) Fix normalized z = 0.0 (bottom).
        3) Map (x_norm, y_norm, 0) into physical coords (mins, maxs).
        4) Take the Cartesian product -> (N²,3).
        
        Args:
            N: Number of points per horizontal axis.
            method: 'linspace', 'uniform', or 'lhs'.
            seed: RNG seed for reproducibility.
            
        Returns:
            Tensor of shape (N**2, 3), dtype=self.mins.dtype, device=self.device,
            with x∈(xmin,xmax), y∈(ymin,ymax), z=z_min.
        """
        # 1) normalized 1D samples
        x_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        y_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        z_norm = torch.zeros(1, device=self.device, dtype=self.mins.dtype)  # normalized bottom
        # 2) map into physical box
        span    = self.maxs - self.mins               # (3,)
        x_phys  = x_norm * span[0] + self.mins[0]     # (N,)
        y_phys  = y_norm * span[1] + self.mins[1]     # (N,)
        z_phys  = z_norm * span[2] + self.mins[2]     # (1,) -> equals self.mins[2]
        # 3) Cartesian product -> (N**2, 3)
        return torch.cartesian_prod(x_phys, y_phys, z_phys)
    
    def generate_random_points_on_front(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N**2 points on the front face (x = x_max) of the 3D box.
        
        1) Sample N values in (0,1) along y and z via your chosen method.
        2) Fix normalized x = 1.0 (front).
        3) Map (1, y_norm, z_norm) into physical coords (mins, maxs).
        4) Take the Cartesian product -> (N**2, 3).
        
        Args:
            N: Number of points per vertical axis.
            method: 'linspace', 'uniform', or 'lhs'.
            seed: RNG seed for reproducibility.
        
        Returns:
            Tensor of shape (N**2, 3), dtype=self.mins.dtype, device=self.device,
            with x = xmax, y∈(ymin,ymax), z∈(zmin,zmax).
        """
        #1) normalized 1D samples
        x_norm = torch.ones(1, device=self.device, dtype=self.mins.dtype)  # normalized front
        y_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        z_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        #2) map into physical box
        span   = self.maxs - self.mins                  # (3,)
        x_phys = x_norm * span[0] + self.mins[0]        # (1,) -> equals self.maxs[0]
        y_phys = y_norm * span[1] + self.mins[1]        # (N,)
        z_phys = z_norm * span[2] + self.mins[2]        # (N,)
        #3) Cartesian product -> (N**2, 3)
        return torch.cartesian_prod(x_phys, y_phys, z_phys)
    
    def generate_random_points_on_back(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N**2 points on the back face (x = x_min) of the 3D box.
        
        1) Sample N values in (0,1) along y and z via your chosen method.
        2) Fix normalized x = 0.0 (back).
        3) Map (0, y_norm, z_norm) into physical coords (mins, maxs).
        4) Take the Cartesian product -> (N**2, 3).
        
        Args:
            N: Number of points per vertical axis.
            method: 'linspace', 'uniform', or 'lhs'.
            seed: RNG seed for reproducibility.
            
        Returns:
            Tensor of shape (N**2, 3) with x = xmin, y∈(ymin,ymax), z∈(zmin,zmax).
        """
        #1) normalized 1D samples
        x_norm = torch.zeros(1, device=self.device, dtype=self.mins.dtype)  # normalized back
        y_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        z_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        #2) map into physical box
        span   = self.maxs - self.mins
        x_phys = x_norm * span[0] + self.mins[0]        # (1,) -> equals self.mins[0]
        y_phys = y_norm * span[1] + self.mins[1]        # (N,)
        z_phys = z_norm * span[2] + self.mins[2]        # (N,)
        #3) Cartesian product -> (N**2, 3)
        return torch.cartesian_prod(x_phys, y_phys, z_phys)
    
    def generate_random_points_on_right(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N**2 points on the right face (y = y_max) of the 3D box.
        
        1) Sample N values in (0,1) along x and z via your chosen method.
        2) Fix normalized y = 1.0 (right).
        3) Map (x_norm, 1, z_norm) into physical coords (mins, maxs).
        4) Take the Cartesian product -> (N**2, 3).
        
        Args:
            N: Number of points per vertical axis.
            method: 'linspace', 'uniform', or 'lhs'.
            seed: RNG seed for reproducibility.
            
        Returns:
            Tensor of shape (N**2, 3) with x∈(xmin,xmax), y = ymax, z∈(zmin,zmax).
        """
        #1) normalized 1D samples
        x_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        y_norm = torch.ones(1, device=self.device, dtype=self.mins.dtype)  # normalized right
        z_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        #2) map into physical box
        span   = self.maxs - self.mins
        x_phys = x_norm * span[0] + self.mins[0]        # (N,)
        y_phys = y_norm * span[1] + self.mins[1]        # (1,) -> equals self.maxs[1]
        z_phys = z_norm * span[2] + self.mins[2]        # (N,)
        #3) Cartesian product -> (N**2, 3)
        return torch.cartesian_prod(x_phys, y_phys, z_phys)
    
    def generate_random_points_on_left(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Generate N**2 points on the left face (y = y_min) of the 3D box.
        
        1) Sample N values in (0,1) along x and z via your chosen method.
        2) Fix normalized y = 0.0 (left).
        3) Map (x_norm, 0, z_norm) into physical coords (mins, maxs).
        4) Take the Cartesian product -> (N**2, 3).
        
        Args:
            N: Number of points per vertical axis.
            method: 'linspace', 'uniform', or 'lhs'.
            seed: RNG seed for reproducibility.
            
        Returns:
            Tensor of shape (N**2, 3) with x∈(xmin,xmax), y = ymin, z∈(zmin,zmax).
        """
        #1) normalized 1D samples
        x_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        y_norm = torch.zeros(1, device=self.device, dtype=self.mins.dtype)  # normalized left
        z_norm = self._generate_a_list_of_points(N=N, method=method, seed=seed)
        #2) map into physical box
        span   = self.maxs - self.mins
        x_phys = x_norm * span[0] + self.mins[0]        # (N,)
        y_phys = y_norm * span[1] + self.mins[1]        # (1,) -> equals self.mins[1]
        z_phys = z_norm * span[2] + self.mins[2]        # (N,)
        #3) Cartesian product -> (N**2, 3)
        return torch.cartesian_prod(x_phys, y_phys, z_phys)
    
class Layered_Contact_Geometry(PINN_Geometry):
    """Stack of axis-aligned bounding boxes (layers) along the z-axis.
    
    Builds `n_layers` contiguous boxes of size (lx, ly, lz), stacked so that
    layer i spans z in [i * lz, (i + 1) * lz]. Offers per-point lookup of which
    layer a given (x, y, z) falls into, plus sampling and graph utilities.
    
    Attributes:
        lx (torch.Tensor): Half-width in x.
        ly (torch.Tensor): Half-width in y.
        lz (torch.Tensor): Thickness of each layer in z.
        n_layers (int): Number of layers.
        bboxes (torch.Tensor): Tensor of shape (L, 2, 3) stacking [mins; maxs].
        layer_mins (torch.Tensor): Tensor of shape (L, 3) of min corners.
        layer_maxs (torch.Tensor): Tensor of shape (L, 3) of max corners.
        standard_layers (List[Single_Cube_Geometry]): Helpers per layer.
    """
    def __init__(self, para: Geometry_Parameter, device: str) -> None:
        """Initialize stacked contact geometry along z.
        
        Args:
            para (Geometry_Parameter): Parameters with fields:
                * lx (float): full width in x-direction.
                * ly (float): full width in y-direction.
                * lz (float): thickness of each layer.
                * n_layers (int): number of layers.
            device (str): Torch device tag (e.g. "cpu", "cuda").
        """
        self.lx = torch.tensor(para.lx / 2.0, device=device)
        self.ly = torch.tensor(para.ly / 2.0, device=device)
        self.lz = torch.tensor(para.lz, device=device)
        self.area = para.lx * para.ly
        self.n_layers = para.n_layers
        self.device = device
        
        self.bboxes = torch.Tensor([[[0.0, 0.0, i * self.lz],
                [self.lx, self.ly, (i + 1) * self.lz]]
                for i in range(self.n_layers)]).to(device) # (L, 2, 3)
        self.layer_mins = self.bboxes[:, 0, :]   # (L, 3)
        self.layer_maxs = self.bboxes[:, 1, :]   # (L, 3)
        standard_bbox = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
        self.standard_layers = [Single_Cube_Geometry(bbox=standard_bbox)
                for i in range(self.n_layers)]
        super().__init__(device=device, dtype=self.bboxes.dtype)
    
    def _calculate_layer_indices(self, x: torch.Tensor) -> torch.LongTensor:
        """Assign each point to the first layer whose top z ≥ point.z.
        
        Points lying exactly on a layer boundary z = k·lz end up in layer k+1.
        
        Args:
            x: Tensor of shape (N,3) or (3,) representing points in physical coords.
            
        Returns:
            LongTensor of shape (N,) with layer indices in [0, n_layers-1].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # common helper: for each point find the layer whose top z ≥ z_point
        z = x[:, 2]                               # [N]
        layer_z_maxs = self.layer_maxs[:, 2]      # [L]
        # mask[n,l] = True if z[n] <= layer_z_maxs[l]
        mask = z.unsqueeze(1) <= layer_z_maxs.unsqueeze(0)
        # first True in each row gives the layer index
        return mask.float().argmax(dim=1).long()  # [N]
    
    def generate_random_points_inside(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample interior points in each layer by axis-wise sampling and Cartesian product.
        
        For each of the `n_layers`, this method samples `N` points along each
        of the x, y, and z axes according to the specified `method` and `seed`,
        then forms the Cartesian product of those 1D samples to get N^3 points
        inside that layer. The results from all layers are concatenated.
        
        Args:
            N (int): Number of points to sample along each axis per layer.
            method (str): Sampling strategy. One of:
                - 'linspace': equally spaced in (0,1) before scaling to the layer.
                - 'uniform': random in [eps,1−eps] before scaling (uses helper eps).
                - 'lhs': Latin-hypercube sampling in (0,1) before scaling.
            seed (int): RNG seed for reproducibility (used by 'uniform' and 'lhs').
            
        Returns:
            torch.Tensor: Tensor of shape (n_layers * N**3, 3) with dtype
                `self.dtype` and device `self.device`, containing the
                sampled interior coordinates for all layers.
        """
        all_pts = []
        for i in range(self.n_layers):
            if method == 'linspace' or method == 'uniform':
                x = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 0] - self.layer_mins[i, 0]) + self.layer_mins[i, 0]
                y = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 1] - self.layer_mins[i, 1]) + self.layer_mins[i, 1]
                z = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 2] - self.layer_mins[i, 2]) + self.layer_mins[i, 2]
            elif method == 'lhs':
                from pyDOE import lhs
                np.random.seed(seed)
                samples = lhs(3, samples=N)  # 3D
                x = torch.tensor(samples[:, 0], dtype=self.lx.dtype, device=self.device) * (
                    self.layer_maxs[i, 0] - self.layer_mins[i, 0]) + self.layer_mins[i, 0]
                y = torch.tensor(samples[:, 1], dtype=self.lx.dtype, device=self.device) * (
                    self.layer_maxs[i, 1] - self.layer_mins[i, 1]) + self.layer_mins[i, 1]
                z = torch.tensor(samples[:, 2], dtype=self.lx.dtype, device=self.device) * (
                    self.layer_maxs[i, 2] - self.layer_mins[i, 2]) + self.layer_mins[i, 2]
            all_pts.append(torch.cartesian_prod(x, y, z))
        return torch.cat(all_pts, dim=0)
    
    def generate_random_points_on_top(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample points on the top face of the last layer.
        
        For the topmost layer only, this samples N points along each of the
        x and y axes using the given method and seed, then fixes z at the
        maximum z-coordinate of the last layer. The Cartesian product of
        x and y yields N**2 points on that face.
        
        Args:
            N (int): Number of points to sample along each axis (x and y).
            method (str): Sampling strategy ('linspace', 'uniform', 'lhs').
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N**2, 3), dtype `self.dtype`,
                device `self.device`, containing (x, y, z_top) coordinates.
        """
        x = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                self.layer_maxs[-1, 0] - self.layer_mins[-1, 0]) + self.layer_mins[-1, 0]
        y = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[-1, 1] - self.layer_mins[-1, 1]) + self.layer_mins[-1, 1]
        z = self.layer_maxs[-1, 2].unsqueeze(0)
        return torch.cartesian_prod(x, y, z)
    
    def generate_random_points_on_bottom(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample points on the bottom face of the bottom layer.
        
        For the bottommost layer only, this samples N points along each of the
        x and y axes using the given method and seed, then fixes z at the
        minimum z-coordinate of layer 0. The Cartesian product of x and y
        yields N**2 points on that face.
        
        Args:
            N (int): Number of points to sample along each axis (x and y).
            method (str): Sampling strategy ('linspace', 'uniform', 'lhs').
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (N**2, 3), dtype `self.dtype`,
                device `self.device`, containing (x, y, z_bottom) coordinates.
        """
        x = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                self.layer_maxs[0, 0] - self.layer_mins[0, 0]) + self.layer_mins[0, 0]
        y = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[0, 1] - self.layer_mins[0, 1]) + self.layer_mins[0, 1]
        z = self.layer_mins[0, 2].unsqueeze(0)
        return torch.cartesian_prod(x, y, z)
    
    def generate_random_points_on_front(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample points on the front face of each layer.
        
        For each of the `n_layers`, this samples N points along the y and z
        axes at the maximum x-coordinate of that layer, using the specified
        method and seed. The Cartesian product of y and z yields N**2 points
        per layer, concatenated across all layers.
        
        Args:
            N (int): Number of points to sample along each axis (y and z).
            method (str): Sampling strategy ('linspace', 'uniform', 'lhs').
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape (n_layers * N**2, 3), dtype `self.dtype`,
                device `self.device`, containing sampled (x_front, y, z) coords.
        """
        all_pts = []
        for i in range(self.n_layers):
            x = self.layer_maxs[i, 0].unsqueeze(0)
            y = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 1] - self.layer_mins[i, 1]) + self.layer_mins[i, 1]
            z = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 2] - self.layer_mins[i, 2]) + self.layer_mins[i, 2]
            all_pts.append(torch.cartesian_prod(x, y, z))
        return torch.cat(all_pts, dim=0)
    
    def generate_random_points_on_back(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample points on the back face of each layer (x = xmin).
        
        For each of the `n_layers`, this samples `N` points along y and z
        at the minimum x-coordinate of that layer, using the given method
        and seed. The Cartesian product of y and z yields N**2 points per
        layer, concatenated across all layers.
        
        Args:
            N (int): Number of samples along each axis (y and z) per layer.
            method (str): Sampling strategy ('linspace', 'uniform', or 'lhs').
            seed (int): RNG seed for reproducibility.
            
        Returns:
            torch.Tensor: Shape `(n_layers * N**2, 3)`, dtype `self.dtype`,
                device `self.device`, containing (x_back, y, z) coords.
        """
        all_pts = []
        for i in range(self.n_layers):
            x = self.layer_mins[i, 0].unsqueeze(0)
            y = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 1] - self.layer_mins[i, 1]) + self.layer_mins[i, 1]
            z = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 2] - self.layer_mins[i, 2]) + self.layer_mins[i, 2]
            all_pts.append(torch.cartesian_prod(x, y, z))
        return torch.cat(all_pts, dim=0)
    
    def generate_random_points_on_right(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample points on the right face of each layer (y = ymax).
        
        For each of the `n_layers`, this samples `N` points along x and z
        at the maximum y-coordinate of that layer, using the given method
        and seed. The Cartesian product of x and z yields N**2 points per
        layer, concatenated across all layers.
        
        Args:
            N (int): Number of samples along each axis (x and z) per layer.
            method (str): Sampling strategy ('linspace', 'uniform', or 'lhs').
            seed (int): RNG seed for reproducibility.
            
        Returns:
            torch.Tensor: Shape `(n_layers * N**2, 3)`, dtype `self.dtype`,
                device `self.device`, containing (x, y_right, z) coords.
        """
        all_pts = []
        for i in range(self.n_layers):
            x = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 0] - self.layer_mins[i, 0]) + self.layer_mins[i, 0]
            y = self.layer_maxs[i, 1].unsqueeze(0)
            z = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 2] - self.layer_mins[i, 2]) + self.layer_mins[i, 2]
            all_pts.append(torch.cartesian_prod(x, y, z))
        return torch.cat(all_pts, dim=0)
    
    def generate_random_points_on_left(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample points on the left face of each layer (y = ymin).
        
        For each of the `n_layers`, this samples `N` points along x and z
        at the minimum y-coordinate of that layer, using the given method
        and seed. The Cartesian product of x and z yields N**2 points per
        layer, concatenated across all layers.
        
        Args:
            N (int): Number of samples along each axis (x and z) per layer.
            method (str): Sampling strategy ('linspace', 'uniform', or 'lhs').
            seed (int): RNG seed for reproducibility.
            
        Returns:
            torch.Tensor: Shape `(n_layers * N**2, 3)`, dtype `self.dtype`,
                device `self.device`, containing (x, y_left, z) coords.
        """
        all_pts = []
        for i in range(self.n_layers):
            x = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 0] - self.layer_mins[i, 0]) + self.layer_mins[i, 0]
            y = self.layer_mins[i, 1].unsqueeze(0)
            z = self._generate_a_list_of_points(N=N, method=method, seed=seed * (
                    self.layer_maxs[i, 2] - self.layer_mins[i, 2]) + self.layer_mins[i, 2])
            all_pts.append(torch.cartesian_prod(x, y, z))
        return torch.cat(all_pts, dim=0)
    
    def generate_random_points_on_sides(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample points on all vertical side faces of each layer.
        
        This concatenates samples from front, back, right, and left faces
        across all `n_layers`. Each face contributes N**2 points per layer.
        
        Args:
            N (int): Number of samples along each axis per face (so each face yields N**2 points per layer).
            method (str): Sampling strategy ('linspace', 'uniform', 'lhs').
            seed (int): RNG seed for reproducibility.
            
        Returns:
            torch.Tensor: Shape `(4 * n_layers * N**2, 3)`, dtype `self.dtype`,
                device `self.device`, containing (x, y, z) coordinates on all sides.
        """
        return torch.cat([
            self.generate_random_points_on_front(N=N, method=method, seed=seed),
            self.generate_random_points_on_back(N=N, method=method, seed=seed),
            self.generate_random_points_on_right(N=N, method=method, seed=seed),
            self.generate_random_points_on_left(N=N, method=method, seed=seed)
        ], dim=0)
    
    def generate_random_points_on_contact(self, N: int, method: str, seed: int) -> torch.Tensor:
        """Sample points on inter-layer contact planes between adjacent layers.
        
        For each interface between layer i and i+1, samples N**2 points
        on the horizontal plane z = layer_maxs[i, 2]. Points are placed
        by sampling x and y via the given strategy and concatenating.
        
        Args:
            N (int): Number of samples along each axis per interface (yields N**2 per interface).
            method (str): Sampling strategy ('linspace', 'uniform', 'lhs').
            seed (int): RNG seed for reproducibility.
            
        Returns:
            torch.Tensor: Shape `((n_layers - 1) * N**2, 3)`, dtype `self.dtype`,
                device `self.device`, containing (x, y, z_contact) coords.
        """
        all_pts = []
        for i in range(self.n_layers - 1):
            x = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 0] - self.layer_mins[i, 0]) + self.layer_mins[i, 0]
            y = self._generate_a_list_of_points(N=N, method=method, seed=seed) * (
                    self.layer_maxs[i, 1] - self.layer_mins[i, 1]) + self.layer_mins[i, 1]
            z = self.layer_maxs[i, 2].unsqueeze(0)
            all_pts.append(torch.cartesian_prod(x, y, z))
        return torch.cat(all_pts, dim=0)
    
    def generate_random_domain_points(self, N_domain: int, N_bc: int, method: str, seed: int = 0) -> torch.Tensor:
        """Sample combined interior and boundary points over the entire geometry.
        
        Combines interior samples with boundary samples on top, bottom,
        sides, and contact planes into one tensor.
        
        Args:
            N_domain (int): Number of interior points to sample per layer (yields n_layers * N_domain**3 total).
            N_bc (int): Number of samples along each axis for each boundary type (yields varied totals per boundary).
            method (str): Sampling strategy ('linspace', 'uniform', 'lhs').
            seed (int): RNG seed for reproducibility.
            
        Returns:
            torch.Tensor: Tensor of shape `(N_domain_total + N_bc_total, 3)`, dtype `self.dtype`,
                device `self.device`, containing all domain and boundary coordinates.
        """
        inside_points = self.generate_random_points_inside(N=N_domain, method=method, seed=seed)
        top_points = self.generate_random_points_on_top(N=N_bc, method=method, seed=seed)
        bottom_points = self.generate_random_points_on_bottom(N=N_bc, method=method, seed=seed)
        side_points = self.generate_random_points_on_sides(N=N_bc, method=method, seed=seed)
        contact_points = self.generate_random_points_on_contact(N=N_bc, method=method, seed=seed)
        domain_points = torch.cat([inside_points, top_points, bottom_points,
                side_points, contact_points], dim=0)
        return domain_points
    
    def generate_contact_graph(self, N: int, method: str, seed: int, k: int) -> Data:
        """Generate a k-NN graph over inter-layer contact-plane samples.
        
        Samples N points on each inter-layer contact plane using the specified
        method and seed, then constructs a k-nearest neighbor graph over all
        sampled contact points.
        
        Args:
            N (int): Number of points to sample per contact plane.
            method (str): Sampling strategy ('linspace', 'uniform', or 'lhs').
            seed (int): RNG seed for reproducibility.
            k (int): Number of nearest neighbors per node.
            
        Returns:
            Data: A torch_geometric Data object with:
                - x (Tensor): Node feature tensor of shape (M, 3) containing sampled contact points.
                - edge_index (LongTensor): Edge index tensor of shape (2, E).
        """
        contact_points = self.generate_random_points_on_contact(N=N, method=method, seed=seed)
        edge_index = knn_graph(x=contact_points, k=k, loop=False)
        data_contact = Data(x=contact_points, edge_index=edge_index)
        return data_contact
    
    def update_geometry(self, step: int, pred_func: callable) -> None:
        """Update bounding boxes by applying a prediction function at boundary points.
        
        Samples a grid of points on the bottom, contact, and top faces of each layer,
        then queries `pred_func` to predict z-offsets. Samples on the right and left
        faces to predict y-offsets, and on the front and back faces to predict x-offsets.
        Updates `self.bboxes` by shifting min and max corners by the mean predicted offsets.
        
        Args:
            step (int): Current training or update step passed to `pred_func`.
            pred_func (callable): Function with signature
                `pred_func(x: Tensor, n_step: int) -> Tensor` that returns
                predicted displacements for input points.
        """
        N = 16
        method = 'linspace'
        seed = 0
        bottom_points = self.generate_random_points_on_bottom(N=N, method=method, seed=seed)
        contact_points = self.generate_random_points_on_contact(N=N, method=method, seed=seed)
        top_points = self.generate_random_points_on_top(N=N, method=method, seed=seed)
        z_points = torch.cat([bottom_points, contact_points, top_points], dim=0)
        pred_z = pred_func(x=z_points, n_step=step)[:, 2]
        
        right_points = self.generate_random_points_on_right(N=N, method=method, seed=seed)
        left_points = self.generate_random_points_on_left(N=N, method=method, seed=seed)
        pred_y_max = pred_func(x=right_points, n_step=step)[:, 1]
        pred_y_min = pred_func(x=left_points, n_step=step)[:, 1]
        
        front_points = self.generate_random_points_on_front(N=N, method=method, seed=seed)
        back_points = self.generate_random_points_on_back(N=N, method=method, seed=seed)
        pred_x_max = pred_func(x=front_points, n_step=step)[:, 0]
        pred_x_min = pred_func(x=back_points, n_step=step)[:, 0]
        for i in range(self.n_layers):
            # (L, 2, 3), 0: min, 1: max
            self.bboxes[i, 0, 0] += pred_x_min[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 1, 0] += pred_x_max[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 0, 1] += pred_y_min[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 1, 1] += pred_y_max[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 0, 2] += pred_z[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 1, 2] += pred_z[(i + 1) * N**2:(i+2) * N**2].mean()
    
    def update_geometry_graph(self, step: int, pred_func: callable) -> None:
        """Update bounding boxes using graph-based predictions at boundary nodes.
        
        Builds k-NN graphs over bottom, contact, and top faces, combines them,
        and applies `pred_func` to predict z-offsets. Builds separate graphs
        on right, left, front, and back faces to predict y- and x-offsets.
        Updates `self.bboxes` by adding mean predicted offsets to each corner.
        
        Args:
            step (int): Current training or update step passed to `pred_func`.
            pred_func (callable): Function with signature
                `pred_func(x: Tensor, edge_index: LongTensor, n_step: int) -> Tensor`
                that returns predicted displacements for graph nodes.
        """
        N = 16
        method = 'linspace'
        seed = 0
        k = 8
        bottom_points = self.generate_bottom_graph(N=N, method=method, seed=seed, k=k)
        contact_points = self.generate_contact_graph(N=N, method=method, seed=seed, k=k)
        top_points = self.generate_top_graph(N=N, method=method, seed=seed, k=k)
        z_points = self.combine_graphs(graph_list=[bottom_points, contact_points, top_points]).to(self.device)
        pred_z = pred_func(x=z_points.x, edge_index=z_points.edge_index, n_step=step)[:, 2]
        
        right_points = self.generate_right_graph(N=N, method=method, seed=seed, k=k).to(self.device)
        left_points = self.generate_left_graph(N=N, method=method, seed=seed, k=k).to(self.device)
        pred_y_max = pred_func(x=right_points.x, edge_index=right_points.edge_index, n_step=step)[:, 1]
        pred_y_min = pred_func(x=left_points.x, edge_index=left_points.edge_index, n_step=step)[:, 1]
        
        front_points = self.generate_front_graph(N=N, method=method, seed=seed, k=k)
        back_points = self.generate_back_graph(N=N, method=method, seed=seed, k=k)
        pred_x_max = pred_func(x=front_points.x, edge_index=front_points.edge_index, n_step=step)[:, 0]
        pred_x_min = pred_func(x=back_points.x, edge_index=back_points.edge_index, n_step=step)[:, 0]
        for i in range(self.n_layers):
            # (L, 2, 3), 0: min, 1: max
            self.bboxes[i, 0, 0] += pred_x_min[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 1, 0] += pred_x_max[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 0, 1] += pred_y_min[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 1, 1] += pred_y_max[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 0, 2] += pred_z[i * N**2:(i+1) * N**2].mean()
            self.bboxes[i, 1, 2] += pred_z[(i + 1) * N**2:(i+2) * N**2].mean()
        