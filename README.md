# Assignment 3 Computer vision (Davide Barbieri, 12871745)

## Dependencies

All the required packages are in a yml environment file. If you have conda install, you can simply do 

`conda env create -f environment.yml`

to install the dependencies. Keep in mind that `cmake` is needed for the installation of `dlib`. Finally, before running the scripts (once per shell) run

`conda activate cv2`

You will also need to download (BFM model)[(https://faces.dmi.unibas.ch/bfm/bfm2017.html] and place `model2017-1_face12_nomouth.h5` in `data`.

## Scripts

### test.py

This script manually constructs the pipeline required for sections 2 (Morphable Model) and 3 (Pinhole camera model). Please run `fit.py --help` to see the arguments available. 

The argument `--up_to` allows to specify at which point to stop the pipeline. 

The latent variable **alpha** and **delta** can be given, by pointing to a serialized tuple whose first element is a tuple (alpha, delta). An example of such file can be found in `latent/test_3D_transformation.pkl`, which can be pass using the argument `--latent`. If such a file is not specified, *alpha* and *delta* are uniformly sampled. The size of these two latent bariables can be specified using `--size_id` and `--size_exp` respectively, or otherwise left to the default *30* and *20* (respectively).

A transformation can be specified via `--omega` (Z-Y-Z Euler angle of rotation) and `--t` (X-Y-Z translation).

Finally, Camera settings can be changed from the defaults by using `--near_far` (by passing two value, one for the near clip, one for the far clip), `--fov` (to pass the angle in degrees of the field of view), and `--aratio` (to specify the aspect ratio).

### fit.py

This script uses a more streamlined version of the aforementioned pipe to fit latent variable **alpha**, **delta**, **omega**, and **t**, from a given image (jpeg or png, other formats have not been tested). Again, please run `python3 fit.py --help` for a full overview of the arguments.

The input image file can be passed as a relative or global path using the argument `--target`. Similarly `--output` specifies the name of the binary file in which to store the fit latent variables, given the target.

`--size_id`, `--size_exp`, `--fov`, `--aratio`, and `--near_far` have the same use as in `test.py`.

Regarding the **hyperparameters**, these can be also set using command line arguments. `--epochs` specifies the maximum amount of epochs to run, `--lr` the learning rate to use, `--reg` (folowed by two values) the lambda1 and lambda2 regularization weights.

A starting value for **omega** and **t** can be given using `--omega` and `--t`.

Finally, if `--plotting` is set to true, the landmark points' fit before and after training will be shown, together with the extracted target landmarks on top of the picture.

### texturing.py

This script takes a file with serialized latent variables (extracted by `fit.py`) and the target image, to then apply texture to said image. Run `python3 tecturing.py --help` to have a full overview of the parameters.

`--target` can be used to specify the path to the image file (used as target), while `--latent` will need to point to the binary file containing the latent variables inferred (*alpha*, *delta*, *omega*, and *t*). `--output` can be used to specify the name of the outout image and 3D mesh (the same name with different extensions will be used).

The remaining parameters can be left to default, and have the same use as in the previous scripts.

### Other Scripts

The other scripts define the pipeline model itself. Starting from `face.py`, here the face multilinear PCA, and face rotation (and translation) are defined. Then `camera.py` defines a differentiable module that apply 2D projection and view-point transformation. `pipeline.py` contains full modules that concatenate the previous one, while also applying data scaling/conversion for compatibility between pipeline's stages.

## Running The code

### Morphable Model (Section 2)

To get a 3D point cloud, from uniformly sampled *alpha* and *delta*, the following command was used multiple times

```
python3 test.py --up_to 3D --face_3D_file ../meshes/face_3D_rand.obj
```

This will generate the object file `meshes/face_3D_rand.obj`.

### Rotation of 3D model (Section 3.1)

The file `latent/test_3D_transform.pkl` contains values for *alpha* and *delta* from a random face. This can be used to apply rotation to the same face and observe the results. The following 3 commands can be used.

```
python3 test.py --latent ../latent/test_3D_transform.pkl --up_to rotate --face_wt_file ../meshes/face_wt_r0.obj --omega 0 0 0 --t 0 0 0;
python3 test.py --latent ../latent/test_3D_transform.pkl --up_to rotate --face_wt_file ../meshes/face_wt_r10.obj --omega 0 10 0 --t 0 0 0;
python3 test.py --latent ../latent/test_3D_transform.pkl --up_to rotate --face_wt_file ../meshes/face_wt_r-10.obj --omega 0 -10 0 --t 0 0 0;
```

These will generate the files `../meshes/face_wt_r0.obj` (face with 0 0 0 rotation), `face_wt_r10.obj` (face with 10 degrees of Oy rotation), and `face_wt_r-10.obj` (face with -10 degrees of Oy rotation).

### Face 2D projection (Section 3.2)

To render the 2D face, you can simply run

```
python3 test.py --omega 0 10 0 --t 0 0 -500
```

This will generate a uv plot of the landmark points (`mehses/face_uv.png`) and a 2D png of the rendered face (`meshes/face_2D.png`). To have a 2D projection of the same face used in section 3.1, simply run

```
python3 test.py --omega 0 10 0 --t 0 0 -500 --latent ../latent/test_3D_transform.pkl
```

### Latent parameters estimation (Section 4)

The script `fit.py` serves the purpose of running only this section. The parameters specification is the same as mentioned in the previous section of this README file. An example of latent parameter estimation is the command

```
python3 fit.py --target ../faces/woman.jpeg --output ../latent/woman.pkl --plotting True --epochs 1000
```

This script will only find the parameters of a given face. The rendering is done in the following section.

### Texturing (Section 5)

The script `texturing.py` can be used for this purpose. For instance, to render (with and without texture) the latent parameters used in the example for the previous section, simply run

```
python3 texturing.py --target ../faces/woman.jpeg --latent ../latent/woman.pkl
```

This will generate 4 files:

* `meshes/face_untextured.obj`, which is the point cloud of the face without using the colors from the targets
* `meshes/face_untextured.png`, which is the rendering of the face without using the colors from the targets
* `meshes/face_textured.obj`, which is the point cloud of the face using the colors from the targets
* `meshes/face_textured.png`, which is the rendering of the face using the colors from the targets
