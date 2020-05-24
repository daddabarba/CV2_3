import numpy as np

class FaceComponent:

    def __init__(self, mean : np.ndarray, var : np.ndarray, E : np.ndarray, n : int):

        self.mean = mean.reshape(-1, 3)
        self.var = var[:n]
        self.E = E.reshape(-1, 3, E.shape[-1])[:,:,:n]

class FaceBasis:

    def __init__(self, id_comp : FaceComponent, exp_comp : FaceComponent):

        self.id_comp = id_comp
        self.exp_comp = exp_comp
