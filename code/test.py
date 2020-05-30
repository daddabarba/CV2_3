from supplemental_code import *

from utils import *
from face import *
from camera import *

import h5py
from matplotlib import pyplot as plt
import numpy as np

from argparse import ArgumentParser

def main(args):

    # Get data file
    dt = h5py.File(args.prior, 'r')

    # Extract data
    face_basis = get_face_basis(dt, args.size_id, args.size_exp)

    # SECTION 2
    print("\nSection 2...")

    # Sample alpha and delta
    print("\tSampling latent variables")
    alpha = np.random.uniform(-1, 1, args.size_id).astype(np.float32)
    delta = np.random.uniform(-1, 1, args.size_exp).astype(np.float32)

    # Generate face from respective alpha and delta
    print("\tGenerating face 3D point-cloud")
    face_3D = face_basis(alpha, delta)

    # Save object for later visualization
    print("\tSaving face data")
    save_obj(
        args.face_3D_file,
        face_3D,
        face_basis.color,
        face_basis.mesh,
    )
    print("\tSaved to ", args.face_3D_file)

    if args.up_to is not None and args.up_to == "3D":
        return

    # SECTION 3
    print("\nSection 3...")
    print("Rotating face")

    # Transform face
    print("\tTransforming face with omega: ", args.omega, " and t: ", args.t)
    face_transform = FaceTransform()
    face_wt = face_transform(face_3D, args.omega, args.t)

    print("\tSaving rotated face data")
    save_obj(
        args.face_wt_file,
        face_wt,
        face_basis.color,
        face_basis.mesh
    )
    print("\tSaved to ", args.face_wt_file)

    if args.up_to is not None and args.up_to == "rotate":
        return

    print("Applying camera projection")

    # Init camera
    print("\tInitializing camera with FOV: ", args.fov, " aspect ratio: ", args.aratio, " near-far clips: ", args.near_far)
    camera = Camera(args.fov, args.aratio, args.near_far)

    # Generate image from face
    print("\tGenerating uv image")
    face_uv = camera(face_wt)

    print("\tNormalizing uv image (z coordinate)")
    uv_normalizer = UVNormalizer()
    face_uv_n = uv_normalizer(face_uv)

    # Extracting landmark points
    print("\tExtracting landmark pointsi from", args.landmarks)
    lmks = get_landmarks(args.landmarks)

    face_lmks = face_uv_n[lmks, :2]

    # Generate image
    plt.scatter(face_lmks[:,0], face_lmks[:, 1])
    plt.axis('equal')
    plt.savefig(args.face_uv_file + ".png", dpi=900)
    print("\tSaved to ", args.face_uv_file + ".png")

    save_obj(
        args.face_uv_file + ".obj",
        face_uv_n,
        face_basis.color,
        face_basis.mesh
    )
    print("\tSaved to ", args.face_uv_file + ".obj")

    if args.up_to is not None and args.up_to == "project":
        return

    print("Rendering 2D image")

    print("\tExtracting pixels")
    face_2D = render(face_uv.numpy(), face_basis.color, face_basis.mesh.astype(np.int))

    plt.imsave(args.face_2D_file, face_2D)

if __name__ == "__main__":

    parser = ArgumentParser()

    # Basic parameters

    parser.add_argument(
        "--up_to",
        type = str,
        default = None,
        help = "If given determines where to stop in the pipeline: 3D, rotate, project"
    )
    parser.add_argument(
        "--prior",
        type = str,
        default = "../data/model2017-1_face12_nomouth.h5",
        help = "Location of h5 dictionary containing face prior knowledge"
    )

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

    # Parameters Section 2

    parser.add_argument(
        "--face_3D_file",
        type = str,
        default = "../meshes/face_3D.obj",
        help = "File in which to save 3D model of face",
    )

    # Parameters Section 3

    parser.add_argument(
        "--omega",
        type = lambda x : float(x)/180 * np.pi,
        nargs = 3,
        default = [0.0, 0.0, 0.0],
        help = "Euler angle in Z-Y-X format for face rotation"
    )

    parser.add_argument(
        "--t",
        type = float,
        nargs = 3,
        default = [0.0, 0.0, -500.0],
        help = "Translation for face transformation"
    )

    parser.add_argument(
        "--fov",
        type = lambda x : float(x)/180*np.pi,
        default = 0.5/180*np.pi,
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
        default = [-300, -700],
        help = "Near far clips z coordinates"
    )

    parser.add_argument(
        "--landmarks",
        type = str,
        default = "../data/Landmarks68_model2017-1_face12_nomouth.anl",
        help = "File from which to get landmarks points"
    )

    parser.add_argument(
        "--face_wt_file",
        type = str,
        default = "../meshes/face_wt.obj",
        help = "File in which to save rotated and translated 3D render of face",
    )

    parser.add_argument(
        "--face_uv_file",
        type = str,
        default = "../meshes/face_uv",
        help = "File in which to save uv render of face (both as png and obj file)",
    )

    parser.add_argument(
        "--face_2D_file",
        type = str,
        default = "../meshes/face_2D.png",
        help = "File in which to save 2D render of face",
    )

    main(parser.parse_args())
