from torch import nn, randn as trand, Tensor
from torch.optim import Adam
from torch.autograd import Variable

from pipeline import *
from face import *
from camera import *

from utils import get_landmarks, im2np, torch_norm, plot_status
from supplemental_code import detect_landmark

import h5py
import sys
import pickle

from argparse import ArgumentParser
from tqdm import tqdm

# Losses

class LandmarkLoss(nn.Module):

    def forward(self, predicted, target):
        return ((predicted-target).norm(p=2)**2).mean()

class RegularizationLoss(nn.Module):

    def __init__(self, lambda_alpha : float, lambda_delta : float):
        super().__init__()

        self.lambda_alpha = lambda_alpha
        self.lambda_delta = lambda_delta

    def forward(self, alpha, delta):
        return self.lambda_alpha * (alpha.norm(p=2)**2) + self.lambda_delta * (delta.norm(p=2)**2)

class FitLoss(nn.Module):

    def __init__(self, pipeline : Pipeline, L_lan : LandmarkLoss, L_reg : RegularizationLoss):
        super().__init__()

        self.pipeline = pipeline

        self.L_lan = L_lan
        self.L_reg = L_reg

    def forward(self, latent, transform, target):

        self.pred = self.pipeline(
            *latent,
            *transform
        )

        return self.L_lan(
            self.pred,
            target
        ) + self.L_reg(
            *latent
        )

def main(args):

    # Get landmarks target points

    target_lmks = []

    for target in args.targets:

        print("Extracting landmarks from target ", target)
        target_img = im2np(target)

        target = Tensor(
            detect_landmark(
                target_img
            )
        )

        if args.plotting:
            plt.imshow(target_img)
            lmks = target.detach().numpy()
            plt.scatter(lmks[:,0], lmks[:,1])

            plt.show()

        target_lmks.append(torch_norm(target*-1))

    # Get full pipeline model
    print("Init pipeline model for rendering")

    pipeline = Pipeline(
        renderer3D = Render3DPipe(
            basis = get_face_basis(
                h5py.File(args.prior, 'r'),
                args.size_id,
                args.size_exp
            ),
            transform = FaceTransform(),
        ),
        rendererUV = RenderUVPipe(
            camera = Camera(
                args.fov,
                args.aratio,
                args.near_far
            ),
            normalizer = UVNormalizer(),
        ),
        lmksPipe = LandmarkPipe(
            landmarks = get_landmarks(args.landmarks)
        )
    )

    # Init Loss module
    print("Constructing full loss end-to-end pipeline")

    loss = FitLoss(
        pipeline = pipeline,
        L_lan = LandmarkLoss(),
        L_reg = RegularizationLoss(
            *args.reg
        )
    )

    # Init random latent variavbles
    print("Init latent variables")

    def init_latent(size):
        return Variable(
            trand(size)*2 - 1,
            requires_grad = True
        )

    def set_latent(val):
        return Variable(
            Tensor(
                np.array(
                    val
                )
            ),
            requires_grad = True
        )

    latent = init_latent(args.size_id), init_latent(args.size_exp)

    transforms = []

    for _ in range(len(args.targets)):
        transforms.append((init_latent(3) if args.omega is None else set_latent(args.omega), init_latent(3) if args.t is None else set_latent(args.t)))

    # Init optimizer

    optim = Adam(latent + tuple(i for transform in transforms for i in transform), lr = args.lr)

    # Fit latent parameters
    print("Starting to fit latent parameters")

    if args.plotting:

        for transform, target in zip(transforms, target_lmks):

            _ = loss(
                latent,
                transform,
                target
            )

            plot_status(
                loss.pred.detach().numpy(),
                target.detach().numpy(),
                title = "Initial Setting"
            )

    epoch_bar = tqdm(range(args.epochs))
    for epoch in epoch_bar:

        # Reset gradients
        optim.zero_grad()

        err_tot = 0
        for transform, target in zip(transforms, target_lmks):

            # Compute loss
            err = loss (
                latent,
                transform,
                target
            )

            # Backpropagate loss
            err.backward()

            err_tot += err.item()

        # Update estimate of latent variables
        optim.step()

        # Display results
        epoch_bar.set_description("err: %.3f"%(err_tot/len(args.targets)))

    if args.plotting:
        for transform, target in zip(transforms, target_lmks):

            _ = loss(
                latent,
                transform,
                target
            )

            plot_status(
                loss.pred.detach().numpy(),
                target.detach().numpy(),
                title = "Final Setting"
            )

    with open(args.output, "wb") as f:
        pickle.dump((latent, transforms), f)

if __name__ == "__main__":

    parser = ArgumentParser()

    # Inputs

    parser.add_argument(
        "--targets",
        type = str,
        nargs = "+",
        help = "Input image to fit"
    )

    # Outputs

    parser.add_argument(
        "--output",
        type = lambda x : x + ".pkl" if not x.endswith(".pkl") else x,
        default = "../latent/fit_latent.pkl",
        help = "Output in which to store latent variables"
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

    # Training parameters

    parser.add_argument(
        "--epochs",
        type = int,
        default = 10,
        help = "Max number of epochs to perform"
    )

    parser.add_argument(
        "--lr",
        type = float,
        default = 0.01,
        help = "Learning rate"
    )

    parser.add_argument(
        "--reg",
        type = float,
        nargs = 2,
        default = [0.1, 0.1],
        help = "In order, regularization strength for alpha and delta"
    )

    parser.add_argument(
        "--omega",
        type = lambda x : float(x)/180 * np.pi if x.lower != "None" else None,
        nargs = 3,
        default = [0.0, 0.0, 0.0],
        help = "Initial value for Euler angle in Z-Y-X format for face rotation (set to None for random init)"
    )

    parser.add_argument(
        "--t",
        type = lambda x : float(x) if x.lower != "None" else None,
        nargs = 3,
        default = [0.0, 0.0, -500.0],
        help = "Initial value for translation for face transformation (set to None for random init)"
    )

    # Other setting

    parser.add_argument(
        "--plotting",
        type = lambda x : x.lower() == "true",
        default = False,
        help = "If set to true it plots the fit of the uv landmark points (at each iteration loop)"
    )

    main(parser.parse_args())
