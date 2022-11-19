import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from pixyz import distributions as dist


class Encoder(dist.Normal):
  """
    q(x_t | I_t) = N(x_t | I_t)
  """
  def __init__(self, input_dim: int=3, output_dim: int=2, act_func_name: str="ReLU"):
    super().__init__(var=["x_t"], cond_var=["I_t"])

    activation_func = getattr(nn, act_func_name)

    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=4, stride=2),
        activation_func(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        activation_func(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
        activation_func(),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
        activation_func(),
    )

    self.loc = nn.Sequential(
        nn.Linear(1024, 2),
    )

    self.scale = nn.Sequential(
        nn.Linear(1024, 2),
        nn.Softplus()
    )

  def forward(self, I_t: torch.Tensor) -> dict:
    feature = self.encoder(I_t)
    B, C, W, H = feature.shape
    feature = feature.reshape((B, C*W*H))
    
    loc = self.loc(feature)*torch.pi
    scale = self.scale(feature)

    return {"loc": loc, "scale": scale}


class Decoder(dist.Normal):
  """
    p(I_t | x_t) = N(I_t | x_t)
  """
  def __init__(self, input_dim: int=2, output_dim: int=3, act_func_name: str="ReLU"):
    super().__init__(var=["I_t"], cond_var=["x_t"])

    activation_func = getattr(nn, act_func_name)

    self.up_size_feature = nn.Sequential(
        nn.Linear(2, 1024),
    )

    self.loc = nn.Sequential(
        nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
        activation_func(),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
        activation_func(),
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
        activation_func(),
        nn.ConvTranspose2d(in_channels=32, out_channels=output_dim, kernel_size=6, stride=2),
        nn.Sigmoid(),
    )

  def forward(self, x_t: torch.Tensor) -> dict:

    feature = self.up_size_feature(x_t)
    B, C = feature.shape
    feature = feature.view((B, 1024, 1, 1))

    loc = self.loc(feature)

    return {"loc": loc, "scale": 0.01}


class Transition(dist.Normal): 
  """
    p(x_t | x_{t-1}, u_{t-1}; v_t) = N(x_t | x_{t-1} + ∆t·v_t, σ^2)
  """
  def __init__(self, delta_time: float=.1):
    super().__init__(var=["x_t"], cond_var=["x_tn1", "v_t"])
 
    self.delta_time = delta_time

  def forward(self, x_tn1: torch.Tensor, v_t: torch.Tensor) -> dict:

    x_t = x_tn1 + self.delta_time * v_t

    return {"loc": x_t, "scale": 0.01}


class Velocity(dist.Deterministic):
  """
    v_t = v_{t-1} + ∆t·(A·x_{t-1} + B·v_{t-1} + C·u_{t-1})
    with  [A, log(−B), log C] = diag(f(x_{t-1}, v_{t-1}, u_{t-1}))
  """
  def __init__(self, delta_time: float=.1):
    super().__init__(var=["v_t"], cond_var=["x_tn1", "v_tn1", "u_tn1"], name="f")

    self.output_coefficient = nn.Sequential(
      nn.Linear(2*3, 2),
      nn.ReLU(),
      nn.Linear(2, 2),
      nn.ReLU(),
      nn.Linear(2, 2),
      nn.ReLU(),
      nn.Linear(2, 6),
      nn.ReLU(),
    )
 
    self.delta_time = delta_time
 
  def forward(self, x_tn1: torch.Tensor, v_tn1: torch.Tensor, u_tn1: torch.Tensor) -> dict:
 
    _input = torch.cat([x_tn1, v_tn1, u_tn1], dim=1)
    coefficient = torch.diag_embed(self.output_coefficient(_input))

    A, B, C = coefficient[:, 0:2, 0:2], -torch.exp(coefficient[:, 2:4, 2:4]), torch.exp(coefficient[:, 4:6, 4:6])

    # Dynamics inspired by Newton's motion equation
    v_t = v_tn1 + self.delta_time * (torch.einsum("ijk,ik->ik", A, x_tn1) + torch.einsum("ijk,ik->ik", B, v_tn1) + torch.einsum("ijk,ik->ik", C, u_tn1))
    
    return {"v_t": v_t}
