# Assignment 3 Computer vision (Davide Barbieri, 12871745)

## Dependencies

All the required packages are in a yml environment file. If you have conda install, you can simply do 

`conda env create -f environment.yml`

to install the dependencies. Keep in mind that `cmake` is needed for the installation of `dlib`. Finally, before running the scripts (once per shell) run

`conda activate cv2`

## Scripts

### test.py

This script manually constructs the pipeline required for sections 2 (Morphable Model) and 3 (Pinhole camera model). Please run `fit.py --help` to see the arguments available. 

## Questions

### Rotation of 3D model (3.1)

```
python3 test.py --latent ../latent/test_3D_transform.pkl --up_to rotate --face_wt_file ../meshes/face_wt_r0.obj --omega 0 0 0
python3 test.py --latent ../latent/test_3D_transform.pkl --up_to rotate --face_wt_file ../meshes/face_wt_r10.obj --omega 0 10 0
python3 test.py --latent ../latent/test_3D_transform.pkl --up_to rotate --face_wt_file ../meshes/face_wt_r-10.obj --omega 0 -10 0
```
