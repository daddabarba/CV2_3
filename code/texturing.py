from torch import Tensor
import numpy as np

from pipeline import *
from face import *
from camera import *

from supplemental_code import detect_landmark
from utils import resize_img_tensor, im2np, torch_norm
from supplemental_code import save_obj
import pickle

from matplotlib import pyplot as plt

from argparse import ArgumentParser

def transformUVBasis(lmks, target_lmks):
    """
    Transform predicted landmarks to same coordinate system of picture (target landmarks)
    """

    min_lmks = np.min(target_lmks, axis=0)
    max_lmks = np.max(target_lmks, axis=0)

    scale = max_lmks - min_lmks

    return resize_img_tensor(lmks, *(scale)) + min_lmks[None]

def interpolate2D(uv, img):
    """
    Performs bilinear interpolation on UV projections to infer color for each point
    """

    x, y = uv.T

    # Get closest points
    x1, y1 = np.floor(uv).T.astype(np.int)
    x2, y2 = np.ceil(uv).T.astype(np.int)

    # Evaluate color function at closest points
    Q11 = img[y1, x1]
    Q12 = img[y1, x2]
    Q21 = img[y2, x1]
    Q22 = img[y2, x2]

    # Interpolate on x

    n = x2-x1
    alpha, beta = ((x2-x)/n)[:, None], ((x-x1)/n)[:, None]

    fxy1 = alpha*Q11 + beta*Q21
    fxy2 = alpha*Q12 + beta*Q22

    # Interpolate on y

    n = y2-y1
    alpha, beta = ((y2-y)/n)[:, None], ((y-y1)/n)[:, None]

    return alpha*fxy1 + beta*fxy2

def main(args):

    # Read image

    print("Extracting targets ... ", end="")

    target_img = im2np(args.target)

    target_lmks = detect_landmark(
        target_img
    )

    print("done")

    # Read latent variables

    print("Loading latent variables values ... ", end="")

    with open(args.latent, "rb") as f:
        latent, transform = pickle.load(f)

    print("done")

    # Init model's Pipeline

    print("Building sparse pipeline model ... ", end="")

    render3D = Render3DPipe(
        basis = get_face_basis(
            h5py.File(args.prior, 'r'),
            args.size_id,
            args.size_exp
        ),
        transform = FaceTransform(),
    )

    renderUV = RenderUVPipe(
        camera = Camera(
            args.fov,
            args.aratio,
            args.near_far
        ),
        normalizer = UVNormalizer(),
    )

    lmksPipe = LandmarkPipe(
        landmarks = get_landmarks(args.landmarks)
    )

    print("done")

    # Get predicted landmarks (in image coordinate system)

    print("Predicting features")

    print("\tPredicting 3D render ... ", end="")
    points3D = render3D(
        *latent, *transform
    ).detach()
    print("done")

    print("\tPredicting UV points ... ", end="")
    pointsUV = renderUV(
        points3D
    ).detach() * -1
    print("done")

    print("\tPredicting landmarks ... ", end="")
    lmks = lmksPipe(
        pointsUV
    ).detach()
    print("done")

    # Transform landmarks to image coordinates system

    print("Showing predicted landmarks on picture coordinates ... ", end="")

    lmks = transformUVBasis(
        lmks,
        target_lmks
    )

    # Plot landmarks on picture

    plt.imshow(target_img)
    plt.scatter(
        lmks[:,0], lmks[:,1],
        label = "pred",
        color = "b"
    )

    plt.show()
    plt.legend

    # Transform UV coordinates in image coordinates

    print("Interpolating colors ... ", end="")

    pointsUV = transformUVBasis(
        pointsUV,
        target_lmks
    )

    color = interpolate2D(
        pointsUV.numpy(),
        target_img
    )

    # Save 3D model

    save_obj(
        "../meshes/test.obj",
        points3D,
        color,
        render3D.basis.mesh
    )

    print("done")

if __name__ == "__main__":

    parser = ArgumentParser()

    # Inputs

    parser.add_argument(
        "--target",
        type = str,
        help = "Input image to fit"
    )

    parser.add_argument(
        "--latent",
        type = lambda x : x + ".pkl" if not x.endswith(".pkl") else x,
        default = "fit_latent.pkl",
        help = "Input which contains latent variables values"
    )

    # Data Parameters

    parser.add_argument(
        "--prior",
        type = str,
        default = "../data/model2017-1_face12_nomouth.h5",
        help = "Location of h5 dictionary containing face prior knowledge"
    )

    parser.add_argument(
        "--landmarks",
        type = str,
        default = "../data/Landmarks68_model2017-1_face12_nomouth.anl",
        help = "File from which to get landmarks points"
    )

    # Basis parameters

    parser.add_argument(
        "--size_id",
        type = int,
        default = 30,
        help = "Number of components for id basis"
    )

    parser.add_argument(
        "--size_exp",
        type = int,
        default = 20,
        help = "Number of components for exp basis"
    )

    # Camera parameters

    parser.add_argument(
        "--fov",
        type = lambda x : float(x)/180*np.pi,
        default = 0.5,
        help = "FOV value"
    )

    parser.add_argument(
        "--aratio",
        type = float,
        default = 2,
        help = "Aspect ratio of view"
    )

    parser.add_argument(
        "--near_far",
        type = float,
        nargs = 2,
        default = [-300, -600],
        help = "Near far clops z coordinates"
    )

    main(parser.parse_args())
