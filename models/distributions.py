import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from pixyz import distributions as dist
from pixyz.utils import epsilon

from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

class Encoder(dist.Normal):
    """
      q(x_t | I_t) = N(x_t | I_t)
    """

    def __init__(self, input_dim: int, output_dim: int, act_func_name: str):
        super().__init__(var=["x_t"], cond_var=["I_t"], name="q")

        activation_func = getattr(nn, act_func_name)

        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.loc = nn.Sequential(
            nn.Linear(1000, output_dim),
        )

        self.scale = nn.Sequential(
            nn.Linear(1000, output_dim),
            nn.Softplus()
        )

    def forward(self, I_t: torch.Tensor) -> dict:
        feature = self.encoder(I_t)
        # B, C, W, H = feature.shape
        # feature = feature.reshape((B, C*W*H))

        loc = self.loc(feature)
        scale = self.scale(feature) + epsilon()

        return {"loc": loc, "scale": scale}


class Decoder(dist.Normal):
    """
      p(I_t | x_t) = N(I_t | x_t)
    """

    def __init__(self, input_dim: int, output_dim: int, act_func_name: str, device: str):
        super().__init__(var=["I_t"], cond_var=["x_t"])

        activation_func = getattr(nn, act_func_name)

        self.up_size_vector = nn.Sequential(
            nn.Linear(2, 1000),
            activation_func(),
        )

        self.loc = nn.Sequential(
            nn.ConvTranspose2d(1000, 512, 3, stride=1),
            activation_func(),
            nn.ConvTranspose2d(512, 256, 6, stride=3),
            activation_func(),
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            activation_func(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            activation_func(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            activation_func(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
        )

        self.device = device

    def forward(self, x_t: torch.Tensor) -> dict:
        batchsize = len(x_t)

        vector = self.up_size_vector(x_t)

        reshaped_vector = vector.reshape((batchsize, 1000, 1, 1))

        loc = self.loc(reshaped_vector)

        return {"loc": loc, "scale": .01}


class Transition(dist.Normal):
    """
      p(x_t | x_{t-1}, u_{t-1}; v_t) = N(x_t | x_{t-1} + ∆t·v_t, σ^2)
    """

    def __init__(self, delta_time: float):
        super().__init__(var=["x_t"], cond_var=["x_tn1", "v_t"])

        self.delta_time = delta_time

    def forward(self, x_tn1: torch.Tensor, v_t: torch.Tensor) -> dict:

        x_t = x_tn1 + self.delta_time * v_t

        return {"loc": x_t, "scale": 0.001}


class Velocity(dist.Deterministic):
    """
      v_t = v_{t-1} + ∆t·(A·x_{t-1} + B·v_{t-1} + C·u_{t-1})
      with  [A, log(−B), log C] = diag(f(x_{t-1}, v_{t-1}, u_{t-1}))
    """

    def __init__(self, batch_size: int, delta_time: float, act_func_name: str, device: str, use_data_efficiency: bool):
        super().__init__(var=["v_t"], cond_var=[
            "x_tn1", "v_tn1", "u_tn1"], name="f")

        activation_func = getattr(nn, act_func_name)
        self.delta_time = delta_time
        self.device = device
        self.use_data_efficiency = use_data_efficiency

        if not self.use_data_efficiency:

            self.coefficient_ABC = nn.Sequential(
                nn.Linear(2*3, 2),
                activation_func(),
                nn.Linear(2, 2),
                activation_func(),
                nn.Linear(2, 2),
                activation_func(),
                nn.Linear(2, 6),
            )

        else:
            self.A = torch.zeros((1, 2, 2)).to(self.device)
            self.B = torch.zeros((1, 2, 2)).to(self.device)
            self.C = torch.diag_embed(torch.ones(1, 2)).to(self.device)

    def forward(self, x_tn1: torch.Tensor, v_tn1: torch.Tensor, u_tn1: torch.Tensor) -> dict:

        combined_vector = torch.cat([x_tn1, v_tn1, u_tn1], dim=1)

        # For data efficiency
        if self.use_data_efficiency:
            A = self.A
            B = self.B
            C = self.C
        else:
            _A, _B, _C = torch.chunk(self.coefficient_ABC(combined_vector), 3, dim=-1)
            A = torch.diag_embed(_A)
            B = -touch.exp(torch.diag_embed(_B))
            C = torch.exp(torch.diag_embed(_C))

        # Dynamics inspired by Newton's motion equation
        v_t = v_tn1 + self.delta_time * (torch.einsum("ijk,ik->ik", A, x_tn1) + torch.einsum(
            "ijk,ik->ik", B, v_tn1) + torch.einsum("ijk,ik->ik", C, u_tn1))

        return {"v_t": v_t}
