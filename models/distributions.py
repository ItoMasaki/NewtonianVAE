import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from pixyz import distributions as dist
from pixyz.utils import epsilon

# import resnet from torchvision
from torchvision import models



class Encoder(dist.Normal):
    """
        q(x_t | I_t, y_t) = Normal(x_t | I_t, y_t)
    """

    def __init__(self, input_dim: int, label_dim: int, output_dim: int, activate_func: str):
        super().__init__(var=["x_t"], cond_var=["I_t", "y_t"], name="q")

        activation_func = getattr(nn, activate_func)

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

        self.loc_coordinate = nn.Sequential(
            nn.Linear(1024 + label_dim, 2),
            # nn.Linear(1024, output_dim),
        )

        self.loc_rot = nn.Sequential(
            nn.Linear(1024 + label_dim, 1),
            nn.Tanh()
        )

        self.scale = nn.Sequential(
            nn.Linear(1024 + label_dim, output_dim),
            # nn.Linear(1024, output_dim),
            nn.Softplus()
        )

    def forward(self, I_t: torch.Tensor, y_t: torch.Tensor) -> dict:
        h = self.encoder(I_t)
        B, C, W, H = h.shape
        h = h.reshape((B, C*W*H))

        h = torch.cat((h, y_t), dim=1)

        # loc = self.loc(h)
        loc_coordinate = self.loc_coordinate(h)
        loc_rot = self.loc_rot(h) * torch.pi

        loc = torch.cat((loc_coordinate, loc_rot), dim=1)

        scale = self.scale(h)

        return {"loc": loc, "scale": scale}


class Decoder(dist.Normal):
    """
        p(I_t | x_t, y_t) = Bernoulli(I_t | x_t, y_t)
    """

    def __init__(self, input_dim: int, label_dim: int, output_dim: int, activate_func: str):
        super().__init__(var=["I_t"], cond_var=["x_t", "y_t"])

        activation_func = getattr(nn, activate_func)

        self.loc = nn.Sequential(
            # nn.Conv2d(input_dim+label_dim+2, 64, 3, stride=1, padding=1),
            nn.Conv2d(input_dim+2, 64, 3, stride=1, padding=1),
            activation_func(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            activation_func(),
            nn.Conv2d(64, output_dim, 3, stride=1, padding=1),
            nn.Tanh()
        )

        self.image_size = 64
        a = np.linspace(-1, 1, self.image_size)
        b = np.linspace(-1, 1, self.image_size)
        x, y = np.meshgrid(a, b)
        x = x.reshape(self.image_size, self.image_size, 1)
        y = y.reshape(self.image_size, self.image_size, 1)
        self.xy = np.concatenate((x, y), axis=-1)

        self.z_dim = input_dim + label_dim
        self.z_dim = input_dim

    def forward(self, x_t: torch.Tensor, y_t: torch.Tensor) -> dict:
        device = x_t.device

        # x_t = torch.cat((x_t, y_t), dim=1)

        batchsize = len(x_t)
        xy_tiled = torch.from_numpy(
            np.tile(self.xy, (batchsize, 1, 1, 1)).astype(np.float32)).to(device)

        # xy_tiled = xy_tiled - coordinate.view(batchsize, 1, 1, 2)
        # xy_tiled = torch.einsum("ijkl,ilm->ijkm", xy_tiled, rotation_matrix)


        z_tiled = torch.repeat_interleave(
            x_t, self.image_size*self.image_size, dim=0).view(batchsize, self.image_size, self.image_size, self.z_dim)

        z_and_xy = torch.cat((z_tiled, xy_tiled), dim=3)
        z_and_xy = z_and_xy.permute(0, 3, 2, 1)

        loc = self.loc(z_and_xy)/2.

        return {"loc": loc, "scale": 1.}


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

    def __init__(self, batch_size: int, delta_time: float, activate_func: str, device: str, use_data_efficiency: bool):
        super().__init__(var=["v_t"], cond_var=[
            "x_tn1", "v_tn1", "u_tn1"], name="f")

        activation_func = getattr(nn, activate_func)
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

        coordinate, rot = torch.chunk(x_tn1, 2, dim=1)
        rot = rot % torch.pi

        v_t = torch.cat((coordinate, rot), dim=1)

        # print(f"v_t: {v_t.detach().cpu().numpy()}", end="\r")

        return {"v_t": v_t}
