import numpy as np

def get_P(bottom_left : tuple, top_right : tuple, n : float, f : float):

    l, b = bottom_left
    r, t = top_right

    return np.array(
        [
            [2*n/(r-l), 0,         (r+l)/(r-l),  0           ],
            [0,         2*n/(t-b), (t+b)/(t-b),  0           ],
            [0,         0,         -(f+n)/(f-n), -2*f*n/(f-n)],
            [0,         0,         -1,           0           ],
        ]
    )
