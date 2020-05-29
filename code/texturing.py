from torch import Tensor

from pipeline import *
from face import *
from camera import *

from utils import resize_img_tensor, im2np, torch_norm
import pickle

from argparse import ArgumentParser

def main(args):

    # Read image

    target_img = Tensor(
        im2np(args.target)
    )

    # Read latent variables

    with open(args.latent, "rb") as f:
        latent, transform = pickle.load(f)


    # Init model's Pipeline

    render3D = Render3DPipe(
        basis = get_face_basis(
            h5py.File(args.prior, 'r'),
            args.size_id,
            args.size_exp
        ),
        transform = FaceTransform(),
    )


    renderUV = RenderUVPipe(
        render3D = render3D,
        camera = Camera(
            args.fov,
            args.aratio,
            args.near_far
        ),
        normalizer = UVNormalizer(),
    )

    pipeline = Pipeline(
        renderer = renderUV,
        landmarks = get_landmarks(args.landmarks)
    )


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
