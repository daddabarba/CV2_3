import numpy as np

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

class FaceTransform:

    def __init__(self, basis : FaceBasis):

        self.basis = basis

    def __get_T(self, omega, t):

        z,y,x = omega

        return np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ])

