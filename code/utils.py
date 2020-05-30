import torch
import numpy as np

from PIL import Image
from supplemental_code import render

from matplotlib import pyplot as plt

def to_homogeneous(x):
    """
    Adds homogeneous coordinate to input if needed

    Parameters:
        x (torch.Tensor) : Nx3 or Nx4 tensor where N is the number of points. If Nx3 a fourth column of ones is added
    """

    if x.shape[1] == 4:
        return x

    return torch.cat([x, torch.ones(x.shape[0], 1)], dim = 1)

def get_landmarks(path : str):
    """
    Returns landmark points array from file

    Parameters:
        path (str) : path to landmark points file
    """

    with open(path, "rt") as f:
        return np.array([int(line) for line in f.readlines()])

def im2np(path : str):
    """
    Converts a given image to numpy array

    Parameters:
        path (str) : path to image file
    """

    return np.asarray(
        Image.open(path)
    )[:, :, :3]

def torch_norm_transform(t):
    """
    Given an NxM image, where N is a number of points, retrurns the transformation that normalizes each axis of the set of points (to be between 0 and 1)

    Parameters:
        t (torch.Tensor) : an NxM (e.g. Nx2) tensor of N (M dimensional) points
    """

    min_t = torch.min(t, dim=0)[0][None]
    max_t = torch.max(t, dim=0)[0][None]

    return 1/(max_t-min_t), -min_t

def torch_norm(t):
    """
    Given an NxM image, where N is a number of points, normalizes each axis of the set of points (to be between 0 and 1)

    Parameters:
        t (torch.Tensor) : an NxM (e.g. Nx2) tensor of N (M dimensional) points
    """

    scale, translation = torch_norm_transform(t)

    return (t+translation)*scale

def get_WH_from_UV(t):
    """
    Returns width and height of image encoded in series of uv coordinates

    Prameters:
        t (torch.Tensor) : an Nx2 set of UV coordinates
    """

    W_t = torch.max(t[:,0]) - torch.min(t[:,0])
    H_t = torch.max(t[:,1]) - torch.min(t[:,1])

    return W_t, H_t

def resize_img_tensor(t, W, H):
    """
    Given a tensor of N points (in uv coordinates) resizes width and height

    Prameters:
        t (torch.Tensor) : an Nx2 set of UV coordinates
        W (int, float) : new width
        H (int, float) : new height
    """

    # first normalize t with axis between 0 and 1

    if H is None:

        #maintain aspect ratio

        W_t, H_t = get_WH_from_UV(t)
        H = int(W* (H_t/W_t))

    t = torch_norm(t)

    # now resize each axis
    t[:,0] *= W
    t[:,1] *= H

    return t

def wrap_render(face_uv, color, mesh):

    face_2D = resize_img_tensor(face_uv, 480, None)
    W_t, H_t  = get_WH_from_UV(face_2D)

    return render(face_2D.numpy(), color, mesh.astype(np.int), int(H_t), int(W_t))

def plot_status(pred, target_lmks, title):
    """
    Compares target with predicted landmarks

    Parameters:
        pred (torch.Tensor) : Nx2 matrix of predicted landmarks
        target_lmks (torch.Tensor) : Nx2 matrix of target landmarks
        title (str) : title of the plot
    """

    plt.figure()

    plt.scatter(pred[:, 0], pred[:, 1], label = "prediction", color = "b")
    plt.scatter(target_lmks[:, 0], target_lmks[:, 1], label = "target", color = "r")

    plt.title(title)
    plt.legend()
    plt.axis('equal')

    plt.show()




