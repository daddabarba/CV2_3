from torch import nn, LongTensor, index_select

from face import *
from camera import *

# Models

class RenderPipe(nn.Module):
    def __init__(self, basis : FaceBasis, transform : FaceTransform, camera : Camera, normalizer : UVNormalizer):
        super().__init__()

        self.basis = basis
        self.transform = transform
        self.camera = camera
        self.normalizer = normalizer

    def forward(self, alpha, delta, omega, t):

        return self.normalizer(
            self.camera(
                self.transform(
                    self.basis(
                        alpha,
                        delta
                    ),
                    omega,
                    t
                )
            )
        )

class Pipeline(nn.Module):

    def __init__(self, renderer : RenderPipe, landmarks : np.ndarray):
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
