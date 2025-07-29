# test/pinn/test_pc_across.py


import torch
import numpy as np
# import deepxde as dde

from typing import Tuple, Dict, Optional, Union, List

# from dl.pinn.pc_across_v1 import Layered_Contact_Geometry, Contact_Boundary_Condition
# from dl.pinn.pc_across_v4 import Layered_Contact_Geometry, Contact_Boundary_Condition, PDE, PINN
from dl.pinn.pc_across_v8 import Layered_Contact_Geometry, Contact_Boundary_Condition

def train_points(geom: Layered_Contact_Geometry, num_domain: int, num_boundary: int) -> np.ndarray:
        X = np.empty((0, geom.dim))
        if num_domain > 0:
            X = geom.random_points(num_domain)
        if num_boundary > 0:
            tmp = geom.random_boundary_points(num_boundary)
            X = np.vstack((tmp, X))
        return X


def self_on_boundary(x, on_boundary, boundary: callable):
    print('DEBUG - [self_on_boundary]')
    results = []
    for i in range(len(x)):
        on_boundary_bool = boundary(x[i], on_boundary[i])
        # print(on_boundary_bool.shape)
        results.append(on_boundary_bool)
    results = np.array(results)
    print(results.shape)
    return results

def collocation_points(geom: Layered_Contact_Geometry, X: np.ndarray, boundary: callable):
    mask = self_on_boundary(x=X, on_boundary=geom.on_boundary(X), boundary=boundary)
    print(mask.shape)
    print(mask)
    print(X.shape)
    X_filtered = X[mask]
    return X_filtered

def bc_points(boundaries: List[callable], train_x_all: np.ndarray, geom: Layered_Contact_Geometry):
    x_bcs = []
    for b in boundaries:
        x_bc = collocation_points(geom=geom, X=train_x_all, boundary=b)
        x_bcs.append(x_bc)
    num_bcs = list(map(len, x_bcs))
    train_x_bc = (np.vstack(x_bcs) if x_bcs else np.empty([0, train_x_all.shape[-1]]))
    return train_x_bc



def test_geometry():
    lx = 45.0
    ly = 45.0
    lz = 15.0
    n_layers = 11
    geom = Layered_Contact_Geometry(lx=lx, ly=ly, lz=lz, n_layers=n_layers)
    cbc = Contact_Boundary_Condition(geom=geom)
    boundaries = [geom.on_contact_interface_boundary]
    bc1 = dde.OperatorBC(geom, cbc.zero_tangential_traction_component1, geom.on_contact_interface_boundary)
    bc2 = dde.OperatorBC(geom, cbc.zero_tangential_traction_component2, geom.on_contact_interface_boundary)
    bc3 = dde.OperatorBC(geom, cbc.zero_complementarity_function_based_fisher_burmeister, geom.on_contact_interface_boundary)
    bcs = [bc1, bc2, bc3]
    
    num_domain = 5
    num_boundary = 5
    # train_next_batch()
    train_x_all = train_points(geom=geom, num_domain=num_domain, num_boundary=num_boundary)
    train_x_bc = bc_points(boundaries=boundaries, train_x_all=train_x_all, geom=geom)

class Test_Layer_Contact_Geometry():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Geometry
        lx = 45.0 / 2.0
        ly = 45.0 # / 2.0
        lz = 15.0
        n_layers = 2 # 11
        self.geom = Layered_Contact_Geometry(lx=lx, ly=ly, lz=lz, n_layers=n_layers, device=self.device)
    
    def test_generating_random_points(self):
        # generate collocation / BC points
        num_domain = 2
        num_bc = 2
        
        points = {
            "Inside": self.geom.generate_random_points_inside(N=num_domain),
            "Top": self.geom.generate_random_points_on_top(N=num_bc),
            "Bottom": self.geom.generate_random_points_on_bottom(N=num_bc),
            "Front": self.geom.generate_random_points_on_front(N=num_bc),
            "Back": self.geom.generate_random_points_on_back(N=num_bc),
            "Right": self.geom.generate_random_points_on_right(N=num_bc),
            "Left": self.geom.generate_random_points_on_left(N=num_bc),
            "Contact": self.geom.generate_random_points_on_contact(N=num_bc)
        }
        # Print verification information
        print("\n" + "="*50)
        print("Generated Points Verification")
        print("="*50)
        
        for name, tensor in points.items():
            # Move to device and enable grad
            tensor = tensor.to(self.device).requires_grad_(True)
            points[name] = tensor  # Update in dict
            
            # Print info
            print(f"\n{name} Points:")
            print(f"Shape: {tensor.shape}")  # Should be (N, 3)
            # print(f"First 3 points:")
            # for i in range(min(3000, len(tensor))):
            #     print(f"  Point {i+1}: x={tensor[i,0].item():.4f}, y={tensor[i,1].item():.4f}, z={tensor[i,2].item():.4f}")
            print(tensor)
            print(f"Requires grad: {tensor.requires_grad}")
            print(f"Device: {tensor.device}")
        
        print("\n" + "="*50)
        print("All points generated successfully!")
        print("="*50)
    
    def test_on_boundaries(self):
        x = torch.tensor([
            [22.5, 7.5,  5.0],
            [0.0,  7.5,  10.0],
            [15.0, 22.5, 5.0],
            [7.5,  0.0,  10.0],
            [7.5,  7.5,  15],
            [22.5, 26.0,  5.0]
        ]).to(self.device)
        
        # Test boundary conditions
        print("\n=== Boundary Test Results ===")
        print(f"Test Points:\n{x}\n")
        
        # Front boundary (y=0)
        on_front = self.geom.are_on_front(x=x)
        print(f"On Front (y=0):\n{on_front}\n")
        
        # Back boundary (y=ly)
        on_back = self.geom.are_on_back(x=x)
        print(f"On Back (y={self.geom.ly}):\n{on_back}\n")
        
        # Right boundary (x=lx)
        on_right = self.geom.are_on_right(x=x)
        print(f"On Right (x={self.geom.lx}):\n{on_right}\n")
        
        # Left boundary (x=0)
        on_left = self.geom.are_on_left(x=x)
        print(f"On Left (x=0):\n{on_left}\n")
        
        # Contact boundaries (between layers)
        on_contact = self.geom.are_on_contact(x=x)
        print(f"On Contact (layer interfaces):\n{on_contact}\n")
        
        # Top boundary (z=lz)
        on_top = self.geom.are_on_top(x=x)
        print(f"On Top (z={self.geom.lz}):\n{on_top}\n")
        
        # Bottom boundary (z=0)
        on_bottom = self.geom.are_on_bottom(x=x)
        print(f"On Bottom (z=0):\n{on_bottom}\n")
        
        # Inside
        inside = self.geom.are_inside(x=x)
        print(f"Inside:\n{inside}\n")
        
        # on boundary
        on_boundary = self.geom.are_on_boundary(x=x)
        print(f"On Boundary:\n{on_boundary}\n")
        
        # Additional check: Point classification
        print("=== Point Classification ===")
        for i, point in enumerate(x):
            boundaries = []
            if on_front[i]: boundaries.append("FRONT")
            if on_back[i]: boundaries.append("BACK")
            if on_right[i]: boundaries.append("RIGHT")
            if on_left[i]: boundaries.append("LEFT")
            if on_contact[i]: boundaries.append("CONTACT")
            if on_top[i]: boundaries.append("TOP")
            if on_bottom[i]: boundaries.append("BOTTOM")
            if inside[i]: boundaries.append("INSIDE")
            if on_boundary[i]: boundaries.append("BOUNDARY")
            
            print(f"Point {i+1} {point.tolist()}: {', '.join(boundaries)}")
    
if __name__ == '__main__':
    # test_geometry()
    lcg = Test_Layer_Contact_Geometry()
    lcg.test_generating_random_points()
    # lcg.test_on_boundaries()
    pass