from utils import *
from face import *

import h5py

from argparse import ArgumentParser

def main(args):

    # Get data file
    dt = h5py.File(args.prior, 'r')

    # Extract data
    face_basis = get_face_basis(dt)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--prior",
        type = str,
        default = "../data/model2017-1_face12_nomouth.h5",
        help = "Location of h5 dictionary containing face prior knowledge"
    )

    main(parser.parse_args())
