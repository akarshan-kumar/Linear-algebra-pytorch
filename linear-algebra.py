import torch
from typing import Annotated,Tuple,Union
from pydantic import BaseModel,field_validator

def dot_product(
        a:torch.Tensor=torch.tensor([1.0,2.0,3.0]), 
        b:torch.Tensor=torch.tensor([4.0,5.0,6.0])) -> None:
    
    
    prod_dot = sum(x*y for x,y in zip(a,b))
    print(type(prod_dot))
    print(f'Python implementation: {prod_dot}')

    prod_dot = torch.dot(a, b)
    print(type(prod_dot))
    print(f'Pytorch implementation: {prod_dot}')

    return prod_dot


def angle_between_vectors(
        a:torch.Tensor=torch.tensor([1.0,2.0,3.0]), 
        b:torch.Tensor=torch.tensor([4.0,5.0,6.0])) -> torch.Tensor:
    
    a_norm = torch.norm(a)
    b_norm = torch.norm(b)
    dot_product = torch.dot(a, b)
    cos_theta = dot_product / (a_norm * b_norm)
    anfle = torch.acos(cos_theta)
    print(f'Angle between vectors (degrees): {torch.rad2deg(anfle):.2f}')
    return anfle

def proj(a:torch.Tensor=torch.tensor([1.0,2.0,3.0]), 
         b:torch.Tensor=torch.tensor([4.0,5.0,6.0])) -> torch.Tensor:
    
    dot_product = torch.dot(a, b)
    bnorm = torch.norm(b)
    projection = (dot_product/bnorm**2) *b
    print(f'Projection of a onto b: {projection}')
    return projection


def point2pointDist(a:torch.Tensor=torch.tensor([1.0,2.0,3.0]), 
                  b:torch.Tensor=torch.tensor([4.0,5.0,6.0]),
                  ) -> torch.Tensor:
    
    dist  = torch.norm(a-b)
    print(f'Point to point distance: {dist:.2f}')
    print(type(dist))
    return dist

def PointTensor(BaseModel):
    point:torch.Tensor

    @field_validator('point')
    def check_shape(cls, v):
        if not isinstance(v, torch.Tensor):
            raise TypeError("Point must be a torch.Tensor")
        if v.shape != (3,):
            raise ValueError("Point must be a 1D tensor of shape (3,), e.g., torch.tensor([x,y,z])")
        return v
    
def LineTensor(BaseModel):
    point:torch.Tensor
    direction:torch.Tensor

    @field_validator('point')
    def check_point_shape(cls, v):
        if not isinstance(v, torch.Tensor):
            raise TypeError("Point must be a torch.Tensor")
        if v.shape != (3,):
            raise ValueError("Point must be a 1D tensor of shape (3,), e.g., torch.tensor([x,y,z])")
        return v
    
    @field_validator('direction')
    def check_direction_shape(cls, v):
        if not isinstance(v, torch.Tensor):
            raise TypeError("Direction must be a torch.Tensor")
        if v.shape != (3,):
            raise ValueError("Direction must be a 1D tensor of shape (3,), e.g., torch.tensor([dx,dy,dz])")
        if torch.norm(v) == 0:
            raise ValueError("Direction vector cannot be zero")
        return v
    
def LineEquation(BaseModel):
    line:torch.Tensor

    @field_validator('line')
    def check_line_shape(cls, v):
        if not isinstance(v, torch.Tensor):
            raise TypeError("Line must be a torch.Tensor")
        if v.shape != (4,):
            raise ValueError("Line must be a 1D tensor of shape (4,), e.g., torch.tensor([x0,y0,z0,dx,dy,dz])")
        return v

def point2lineDist(*args:torch.Tensor)->torch.Tensor:

    if len(args) == 3:
        if not all(isinstance(arg, torch.Tensor) for arg in args) and not all(arg.shape == (3,) for arg in args):
            raise ValueError("Both arguments must be 1D tensors of shape (3,), e.g., torch.tensor([x,y,z])")
        
        distance = torch.norm(torch.linalg.cross(args[0]-args[1],args[2]))/torch.norm(args[2])
        print(f'Point to line distance: {distance:.2f}')
    
    elif len(args) == 2:
        if not all(isinstance(arg, torch.Tensor)for arg in args) and not args[1].shape == (4,) and not args[0].shape == (3,):
            raise ValueError("First argument must be a 1D tensor of shape (3,), e.g., torch.tensor([x,y,z]) and second argument must be a 1D tensor of shape (4,), e.g., torch.tensor([x0,y0,z0,dx,dy,dz])")
        a, b, c, d = args[1]
        numerator = torch.abs(a*args[0][0] + b*args[0][1] + c*args[0][2] + d)
        denominator = torch.sqrt(a**2 + b**2 + c**2)
        print( numerator / denominator)

def line2lineDist(a0:torch.Tensor=torch.tensor([1.0,2.0,3.0]), 
                  a_dir:torch.Tensor=torch.tensor([4.0,5.0,6.0]),
                  b0:torch.Tensor=torch.tensor([7.0,8.0,9.0]),
                  b_dir:torch.Tensor=torch.tensor([10.0,11.0,12.0]),
                  ) -> torch.Tensor:
    n = torch.cross(a_dir, b_dir)   
    n_norm = torch.norm(n)  
    if n_norm == 0:
        denominator = torch.norm(a_dir)
    else:
        denominator = n_norm
    distance = torch.abs(torch.dot(n, b0 - a0)) / denominator
    print(f'Line to line distance: {distance:.2f}')
    return distance


def line2planeDist(line_point:torch.Tensor=torch.tensor([1.0,2.0,3.0]), 
                   line_dir:torch.Tensor=torch.tensor([4.0,5.0,6.0]),
                   point_on_plane:torch.Tensor=torch.tensor([5.0,8.0,7.0]),
                   normal_to_plane:torch.Tensor=torch.tensor([5.0,8.0,7.0])
                   ) -> torch.Tensor:
    
    numerator = torch.abs(torch.dot(normal_to_plane, line_point - point_on_plane))
    denominator = torch.norm(torch.linalg.cross(normal_to_plane, line_dir))
    distance = numerator / denominator
    print(f'Line to plane distance: {distance:.2f}')
    
    return distance

def plane2planeDist(plane1_point:torch.Tensor=torch.tensor([1.0,2.0,3.0]), 
                   plane1_normal:torch.Tensor=torch.tensor([4.0,5.0,6.0]),
                   plane2_point:torch.Tensor=torch.tensor([7.0,8.0,9.0]),
                   plane2_normal:torch.Tensor=torch.tensor([10.0,11.0,12.0]),
                   ) -> torch.Tensor:           
    cross_norm = torch.norm(torch.linalg.cross(plane1_normal, plane2_normal))
    if cross_norm == 0:         
        distance = torch.abs(torch.dot(plane1_normal, plane2_point - plane1_point)) / torch.norm(plane1_normal)
    else:
        distance = 0.0

    print(f'Plane to plane distance: {distance:.2f}')
    return distance


def PCA_with_eigen_values(x:torch.Tensor=torch.tensor([[2.5, 2.4],
                  [0.5, 0.7],
                  [2.2, 2.9],
                  [1.9, 2.2],
                  [3.1, 3.0],
                  [2.3, 2.7]]))-> torch.Tensor:
    
    x_mean = torch.mean(x,dim=0)
    x_centered = x - x_mean
    cov_mat = torch.cov(x_centered.T)
    eigvals, eigvecs = torch.linalg.eig(cov_mat)  # complex output
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    # print(eigvecs)
    sorted_idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]
    k = 2
    W = eigvecs[:, :k]   # top-k eigenvectors
    Z = torch.matmul(x_centered, W)  # projected data


    print(Z)
    return Z

def PCA_with_SVD(x:torch.Tensor=torch.tensor([[2.5, 2.4],
                  [0.5, 0.7],
                  [2.2, 2.9],
                  [1.9, 2.2],
                  [3.1, 3.0],
                  [2.3, 2.7]]))-> torch.Tensor:
    
    x_mean = torch.mean(x,dim=0)
    x_centered = x - x_mean
    U, S, Vt = torch.linalg.svd(x_centered, full_matrices=False)
    k = 2
    W = Vt.T[:, :k]   # top-k right singular vectors
    Z = torch.matmul(x_centered, W)  # projected data

    print(Z)
    return Z

if __name__ == '__main__':
    # P = torch.tensor([1.0, 2.0, 3.0])                # point
    # plane = torch.tensor([2.3, 4.5, 5.6, 67.0]) 
    # point2lineDist(P,plane)
    print(line2planeDist())