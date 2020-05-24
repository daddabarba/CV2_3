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

        R = np.array([
            [1, 0,         0         ],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x) ]
        ]) @ np.array([
            [np.cos(y),  0 , np.sin(y)]
            [0,          1,  0        ],
            [-np.sin(y), 0,  np.cos(y)]
        ]) @ np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z),  0],
            [0,         0,          1]
        ])

        return np.r_[
            np.c_[
                R, t
            ],
            np.array([0,0,0,1])
        ]

    def __to_homogeneous(x):
        return np.c_[x, np.ones(x.shape[0], 1)]

    def __call__(self, omega, t, alpha, delta):

        # Apply transformation matrix
        return self.__get_T(omega, t) @ self.__to_homogeneous(self.basis(alpha, delta))

