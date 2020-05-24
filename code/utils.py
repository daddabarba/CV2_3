import h5py
import numpy as np

from face import *

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

def get_face_basis(dt : h5py._hl.files.File):
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
            n = 30,
        ),
        get_latent_descriptors(
            dt = dt,
            loc = "expression/model",
            n = 20,
        ),
        np.asarray(
            dt["shape/representer/cells"]
        ).T
    )
