from torch import Tensor
import numpy as np

from pipeline import *
from face import *
from camera import *

from supplemental_code import detect_landmark
from utils import resize_img_tensor, im2np
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

    scale = (max_lmks - min_lmks)[None]

    return lmks*scale + min_lmks[None]

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

    targets = []
    target_imgs = []
    for target in args.targets:

        target_imgs.append(
            im2np(
                target
            )
        )

        targets.append(
            detect_landmark(
                target_imgs[-1]
            )
        )

    print("done")

    # Read latent variables

    print("Loading latent variables values ... ", end="")

    with open(args.latent, "rb") as f:
        (alpha, deltas), transforms = pickle.load(f)

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
        landmarks = get_landmarks(args.landmarks),
        norm = False
    )

    print("done")

    # Get predicted landmarks (in image coordinate system)

    for i, (delta, target_img, transform, target_lmks) in enumerate(zip(deltas, target_imgs, transforms, targets)):

        latent = (alpha, delta)

        print("Texturing ", args.targets[i])

        print("Predicting features")

        print("\tPredicting 3D render ... ", end="")
        points3D = render3D(
            *latent,
            *transform
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

        # Get normalization transform to normalize lmks and pointsUV in the same basis
        print("Normalization ... ", end="")

        scale, t = torch_norm_transform(lmks)

        lmks = (lmks+t)*scale
        pointsUV = (pointsUV+t)*scale

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

        # Apply new transformation

        if args.omega is not None:
            transform = Tensor(args.omega), transform[1]

        if args.t is not None:
            transform = transfrom[0], Tensor(args.t)

        points3D = render3D(
            *latent,
            *transform
        ).detach()

        # Save 3D model

        suffix = "" if len(args.targets)==1 else "_"+str(i)

        save_obj(
            args.pointcloud + suffix + ".obj",
            points3D,
            render3D.basis.color,
            render3D.basis.mesh
        )

        # Save textured 3D model

        save_obj(
            args.output + suffix + ".obj",
            points3D,
            color,
            render3D.basis.mesh
        )

        face_uv = Camera(args.fov, args.aratio, args.near_far)(points3D)

        plt.imsave(
            args.output + suffix + ".png",
            wrap_render(
                face_uv,
                color/255,
                render3D.basis.mesh
            ),
        )

        print("done")

if __name__ == "__main__":

    parser = ArgumentParser()

    # Inputs

    parser.add_argument(
        "--targets",
        type = str,
        nargs = "+",
        help = "Input image to fit"
    )

    parser.add_argument(
        "--latent",
        type = lambda x : x + ".pkl" if not x.endswith(".pkl") else x,
        default = "../latent/fit_latent.pkl",
        help = "Input which contains latent variables values"
    )

    # Output

    parser.add_argument(
        "--pointcloud",
        type = str,
        default = "../meshes/face_untextured",
        help = "Location in which to save .obj file of the face without texture (using provided colors). The same name (.png) is used for the rendered version of the PC"
    )

    parser.add_argument(
        "--output",
        type = str,
        default = "../meshes/face_texture",
        help = "Location in which to save .obj file.  The same name (.png) is used for the rendered version of the PC"
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

    # Transformation parameters

    parser.add_argument(
        "--omega",
        type = lambda x : float(x)/180 * np.pi if x.lower != "None" else None,
        nargs = 3,
        default = None,
        help = "Initial value for Euler angle in Z-Y-X format for face rotation (set to None for random init)"
    )

    parser.add_argument(
        "--t",
        type = lambda x : float(x) if x.lower != "None" else None,
        nargs = 3,
        default = None,
        help = "Initial value for translation for face transformation (set to None for random init)"
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
