from torch import nn, Tensor, cat, sin, cos

import numpy as np
import h5py

from utils import *

# Face basis classes

class FaceComponent(nn.Module):

    def __init__(self, mean : np.ndarray, var : np.ndarray, E : np.ndarray, n : int):
        super().__init__()

        self.mean = Tensor(mean.reshape(-1, 3))
        self.var = Tensor(np.sqrt(var[:n]))
        self.E = Tensor(E.reshape(-1, 3, E.shape[-1])[:,:,:n])

    def forward(self, z):
        return self.mean + self.E @ (self.var * z)

class FaceBasis(nn.Module):

    def __init__(self, id_comp : FaceComponent, exp_comp : FaceComponent, mesh : np.ndarray, color : np.ndarray):
        super().__init__()

        self.id_comp = id_comp
        self.exp_comp = exp_comp

        self.mesh = mesh
        self.color = color

    def forward(self, alpha, delta):

        return self.id_comp(alpha) + self.exp_comp(delta)

# Face basis constructors from h5 dictionaries

def get_latent_descriptors(dt : h5py._hl.files.File, loc : str, n : int):
    """
    Extracts mean, pca components, and variance for a given component

    Parameters:
        dt (h5py._hl.files.File) : h5 dictionary containing prior knowledge of object type
        loc (str) : location of component (e.g. expression or face id)
        n (int) : number of components to take

    Returns:
        component (face.FaceComponent) : component encoding mean, variance, and max. variance basis (PCA) for given component
    """

    if loc.endswith('/'):
        loc = loc[:-1]

    mean = np.asarray(dt[loc + '/mean' ], dtype = np.float32)
    var = np.asarray(dt[loc + '/pcaVariance' ], dtype = np.float32)
    E = np.asarray(dt[loc + '/pcaBasis' ], dtype = np.float32)

    return FaceComponent(mean, var, E, n)

def get_face_basis(dt : h5py._hl.files.File, size_id, size_exp):
    """
    Gets full face basis, assuming two components (id and expression)

    Parameters:
        dt (h5py._hl.files.File) : h5 dictionary containing all prior knowledge of face components

    Returns:
        face (face.FaceBasis) : basis for face objects with id and expression components
    """

    return FaceBasis(
        get_latent_descriptors(
            dt = dt,
            loc = "shape/model",
            n = size_id,
        ),
        get_latent_descriptors(
            dt = dt,
            loc = "expression/model",
            n = size_exp,
        ),
        np.asarray(
            dt["shape/representer/cells"],
            dtype = np.float32
        ).T,
        np.asarray(
            dt["color/model/mean"],
            dtype = np.float32
        ).reshape(-1, 3)
    )

# Face transformations


class FaceTransform(nn.Module):

    def __get_T(self, omega, t):

        if not isinstance(omega, Tensor):
            omega = Tensor(omega)

        if not isinstance(t, Tensor):
            t = Tensor(t)

        z,y,x = omega

        R = Tensor([
            [1, 0,      0      ],
            [0, cos(x), -sin(x)],
            [0, sin(x), cos(x) ]
        ]) @ Tensor([
            [cos(y),  0,  sin(y)],
            [0,       1,  0     ],
            [-sin(y), 0,  cos(y)]
        ]) @ Tensor([
            [cos(z), -sin(z), 0],
            [sin(z), cos(z),  0],
            [0,      0,       1]
        ])


        return cat([
            cat([
                R, t[:, None]
            ], dim=1),
            Tensor([[0,0,0,1]])
        ], dim=0)

    def forward(self, x, omega, t):

        # Apply transformation matrix
        return (to_homogeneous(x) @ self.__get_T(omega, t).T)[:, :3]

