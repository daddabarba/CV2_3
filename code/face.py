import numpy as np
import h5py

from utils import to_homogeneous

# Face basis classes

class FaceComponent:

    def __init__(self, mean : np.ndarray, var : np.ndarray, E : np.ndarray, n : int):

        self.mean = mean.reshape(-1, 3)
        self.var = np.sqrt(var[:n])
        self.E = E.reshape(-1, 3, E.shape[-1])[:,:,:n]

    def __call__(self, z):
        return self.mean + self.E @ (self.var * z)

class FaceBasis:

    def __init__(self, id_comp : FaceComponent, exp_comp : FaceComponent, mesh : np.ndarray, color : np.ndarray):

        self.id_comp = id_comp
        self.exp_comp = exp_comp

        self.mesh = mesh
        self.color = color

    def __call__(self, alpha, delta):

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

def __get_T(omega, t):

    z,y,x = omega

    R = np.array([
        [1, 0,         0         ],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x) ]
    ]) @ np.array([
        [np.cos(y),  0 , np.sin(y)],
        [0,          1,  0        ],
        [-np.sin(y), 0,  np.cos(y)]
    ]) @ np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z),  0],
        [0,         0,          1]
    ])


    return np.r_[
        np.c_[
            R, np.array(t)[:, None]
        ],
        np.array([[0,0,0,1]])
    ]

def FaceTransform(x, omega, t):

    # Apply transformation matrix
    return (to_homogeneous(x) @ __get_T(omega, t).T)[:, :3]

