from torch import nn, rand as trand
from torch.autograd import Variable

from face import *
from camera import *

import h5py

from argparse import ArgumentParser

class Pipeline(nn.Module):
    def __init__(self, basis : FaceBasis, transform : FaceTransform, camera : Camera):
        super().__init__()

        self.basis = basis
        self.transform = transform
        self.camera = camera

    def forward(self, alpha, delta, omega, t):

        return self.camera(
            self.transform(
                self.basis(
                    alpha,
                    delta
                ),
                omega,
                t
            )
        )


def main(args):

    # Get full pipeline model

    pipeline = Pipeline(
        basis = get_face_basis(
            h5py.File(args.prior, 'r'),
            args.size_id,
            args.size_exp
        ),
        transform = FaceTransform(),
        camera = Camera(
            args.fov,
            args.aratio,
            args.near_far
        )
    )

    # Init random latent variavbles
    def init_latent(size):
        return Variable(
            trand(size)*2 - 1,
            requires_grad = True
        )

    alpha, delta, omega, t = init_latent(args.size_id), init_latent(args.size_exp), init_latent(3), init_latent(3)

    pred = pipeline(alpha, delta, omega, t)

if __name__ == "__main__":

    parser = ArgumentParser()

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
        default = 60/180*np.pi,
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
        default = [-10, 10],
        help = "Near far clops z coordinates"
    )

    main(parser.parse_args())
