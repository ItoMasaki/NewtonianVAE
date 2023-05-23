import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from pixyz import distributions as dist
from pixyz.utils import epsilon



class Encoder(dist.Normal):
    """
        q(x_t | I_t, y_t) = Normal(x_t | I_t, y_t)
    """

    def __init__(self, input_dim: int, label_dim: int, output_dim: int, act_func_name: str):
        super().__init__(var=["x_t"], cond_var=["I_t", "y_t"], name="q")

        activation_func = getattr(nn, act_func_name)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            activation_func(),
            nn.Conv2d(32, 64, 4, stride=2),
            activation_func(),
            nn.Conv2d(64, 128, 4, stride=2),
            activation_func(),
            nn.Conv2d(128, 256, 4, stride=2),
            activation_func(),
        )

        self.loc = nn.Sequential(
            nn.Linear(1024 + label_dim, 512),
            activation_func(),
            nn.Linear(512, output_dim),
        )

        self.scale = nn.Sequential(
            nn.Linear(1024 + label_dim, 512),
            activation_func(),
            nn.Linear(512, output_dim),
            nn.Softplus()
        )

    def forward(self, I_t: torch.Tensor, y_t: torch.Tensor) -> dict:
        h = self.encoder(I_t)

        B, C, W, H = h.shape
        h = h.reshape((B, C*W*H))

        h = torch.cat((h, y_t), dim=1)

        loc = self.loc(h)
        scale = self.scale(h)

        return {"loc": loc, "scale": scale}


class Decoder(dist.Normal):
    """
        p(I_t | x_t, y_t) = Normal(I_t | x_t, y_t)
    """

    def __init__(self, input_dim: int, label_dim: int, output_dim: int, act_func_name: str):
        super().__init__(var=["I_t"], cond_var=["x_t", "y_t"])

        activation_func = getattr(nn, act_func_name)

        self.up_size_vector = nn.Sequential(
            nn.Linear(input_dim + label_dim, 1024),
            activation_func(),
        )

        self.loc = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            activation_func(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            activation_func(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            activation_func(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Tanh(),
        )

        self.z_dim = input_dim + label_dim

    def forward(self, x_t: torch.Tensor, y_t: torch.Tensor) -> dict:

        x_t = torch.cat((x_t, y_t), dim=1)

        h = self.up_size_vector(x_t).view(-1, 1024, 1, 1)

        loc = self.loc(h)/2.

        return {"loc": loc, "scale": 1.}

    def build_rotation_matrix(self, theta):
        return torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1),
                            torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)], dim=1)


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
            self.A = torch.zeros((1, 3, 3)).to(self.device)
            self.B = torch.zeros((1, 3, 3)).to(self.device)
            self.C = torch.diag_embed(torch.ones(1, 3)).to(self.device)

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
            B = torch.diag_embed(-torch.exp(_B))
            C = torch.diag_embed(torch.exp(_C))

        # Dynamics inspired by Newton's motion equation
        v_t = v_tn1 + self.delta_time * (torch.einsum("ijk,ik->ik", A, x_tn1) + torch.einsum(
            "ijk,ik->ik", B, v_tn1) + torch.einsum("ijk,ik->ik", C, u_tn1))

        # split vector 2 dim and 1 dim using torch.chunk
        # pos, rot = torch.chunk(v_t, 2, dim=1)
        # v_t = torch.cat([pos, torch.cos(rot)], dim=1)

        return {"v_t": v_t}


class LabelEncoder(dist.Categorical):
    def __init__(self, label_dim: int, activation_func_name: str):
        super().__init__(var=["y_t"], cond_var=["I_t"])

        activation_func = getattr(nn, activation_func_name)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            activation_func(),
            nn.Conv2d(32, 64, 4, stride=2),
            activation_func(),
            nn.Conv2d(64, 128, 4, stride=2),
            activation_func(),
            nn.Conv2d(128, 256, 4, stride=2),
            activation_func(),
        )

        self.output_probs = nn.Sequential(
            nn.Linear(1024, label_dim),
            nn.Sigmoid()
        )

    def forward(self, I_t: torch.Tensor) -> dict:
        h = self.encoder(I_t)
        h = h.reshape((h.shape[0], 1024))
        probs = self.output_probs(h)
        return {"probs": probs}
