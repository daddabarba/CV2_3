from torch import nn, LongTensor, index_select

from face import *
from camera import *

# Models

class Render3DPipe(nn.Module):

    def __init__(self, basis : FaceBasis, transform : FaceTransform):
        super().__init__()

        self.basis = basis
        self.transform = transform

    def forward(self, alpha, delta, omega, t):

        return self.transform(
            self.basis(
                alpha,
                delta
            ),
            omega,
            t
        )

class RenderUVPipe(nn.Module):
    def __init__(self, camera : Camera, normalizer : UVNormalizer):
        super().__init__()

        self.camera = camera
        self.normalizer = normalizer

    def forward(self, points3D):

        return self.normalizer(
            self.camera(
                points3D
            )
        )[:, :2]

class LandmarkPipe(nn.Module):

    def __init__(self , landmarks : np.ndarray):
        super().__init__()

        self.landmarks = LongTensor(landmarks)

    def forward(self, pointsUV):

        return torch_norm(
            index_select(
                pointsUV,
                dim = 0,
                index = self.landmarks
            )
        )

class Pipeline(nn.Module):

    def __init__(self, renderer3D : Render3DPipe, rendererUV : RenderUVPipe, lmksPipe : LandmarkPipe):
        super().__init__()

        self.renderer3D = renderer3D
        self.rendererUV = rendererUV
        self.lmksPipe = lmksPipe

    def forward(self, *args):

        return self.lmksPipe(
            self.rendererUV(
                self.renderer3D(
                    *args
                )
            )
        )
