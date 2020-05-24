from supplemental_code import *
from utils import *
from face import *

import h5py

from argparse import ArgumentParser

def main(args):

    # Get data file
    dt = h5py.File(args.prior, 'r')

    # Extract data
    face_basis = get_face_basis(dt)

    # SECTION 2
    print("Section 2...\n")

    # Sample alpha and delta
    print("Sampling latent variables")
    alpha = np.random.rand(30)*2 - 1
    delta = np.random.rand(20)*2 - 1

    # Generate face from respective alpha and delta
    print("Generating face 3D point-cloud")
    random_face = face_basis(alpha, delta)

    # Save object for later visualization
    print("Saving face data")
    save_obj(
        args.rand_face_file,
        random_face,
        np.ones_like(random_face),
        face_basis.mesh
    )

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--prior",
        type = str,
        default = "../data/model2017-1_face12_nomouth.h5",
        help = "Location of h5 dictionary containing face prior knowledge"
    )

    parser.add_argument(
        "--rand_face_file",
        type = str,
        default = "rand_face",
        help = "File in which to save 3D model of face",
    )

    main(parser.parse_args())
