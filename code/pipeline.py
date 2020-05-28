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
    def __init__(self, render3D : Render3DPipe, camera : Camera, normalizer : UVNormalizer):
        super().__init__()

        self.render3D = render3D
        self.camera = camera
        self.normalizer = normalizer

    def forward(self, *args):

        return self.normalizer(
            self.camera(
                self.render3D(
                    *args
                )
            )
        )

class Pipeline(nn.Module):

    def __init__(self, renderer : RenderUVPipe, landmarks : np.ndarray):
        super().__init__()

        self.renderer = renderer
        self.landmarks = LongTensor(landmarks)

    def forward(self, alpha, delta, omega, t):

        return torch_norm(
            index_select(
                self.renderer(
                    alpha,
                    delta,
                    omega,
                    t
                )[:, : 2],
                dim = 0,
                index = self.landmarks
            )
        )
