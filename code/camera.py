import numpy as np
from utils import *

def get_P(bottom_left : tuple, top_right : tuple, near_far : tuple):

    l, b = bottom_left
    r, t = top_right
    n, f = near_far

    return np.array([
        [2*n/(r-l), 0,         (r+l)/(r-l),  0           ],
        [0,         2*n/(t-b), (t+b)/(t-b),  0           ],
        [0,         0,         -(f+n)/(f-n), -2*f*n/(f-n)],
        [0,         0,         -1,           0           ],

    ])

def get_V(bottom_left : tuple, top_right : tuple):

    l, b = bottom_left
    r, t = top_right

    return np.array([
        [(r-l)/2, 0,       0,   (r+l)/2],
        [0,       (t-b)/2, 0,   (t+b)/2],
        [0,       0,       1/2, 1/2    ],
        [0,       0,       0,   1      ]
    ])

class Camera:

    def __init__(self, bottom_left, top_right, near_far):

        self.P = get_P(bottom_left, top_right, near_far)
        self.V = get_V(bottom_left, top_right)

    def __call__(self, x):

        # Go to homogenous
        x = to_homogeneous(x)

        # Apply projection
        x =  x @ self.P.T

        # Apply viewpoint
        x =  x @ self.V.T

        # Normalize depth
        x /= x[:, 2:3]

        # Make homogeneous
        x /= x[:, 3:4]

        return x[:, :2]

