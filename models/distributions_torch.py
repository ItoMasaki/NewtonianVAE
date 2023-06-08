import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal



class Encoder(nn.Module):
    """
        q(x_t | I_t, y_t) = Normal(x_t | I_t, y_t)
    """

    def __init__(self, input_dim: int, label_dim: int, output_dim: int, activate_func: str):
        super().__init__()

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

        self.loc = nn.Sequential(
            nn.Linear(1024 + label_dim, 512),
            activation_func(),
            nn.Linear(512, 3),
        )

        self.scale = nn.Sequential(
            nn.Linear(1024 + label_dim, 512),
            activation_func(),
            nn.Linear(512, 3),
            nn.Softplus()
        )

    def forward(self, I_t: torch.Tensor, y_t: torch.Tensor) -> dict:
        h = self.encoder(I_t)
        B, C, W, H = h.shape
        h = h.reshape((B, C*W*H))

        h = torch.cat((h, y_t), dim=1)

        loc = self.loc(h)
        scale = self.scale(h)

        return Normal(loc, scale).rsample()


class Decoder(nn.Module):
    """
        p(I_t | x_t, y_t) = Bernoulli(I_t | x_t, y_t)
    """

    def __init__(self, input_dim: int, label_dim: int, output_dim: int, activate_func: str):
        super().__init__()

        activation_func = getattr(nn, activate_func)

        self.loc = nn.Sequential(
            nn.Conv2d(input_dim+label_dim+2, 64, 3, stride=1, padding=1),
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

    def forward(self, x_t: torch.Tensor, y_t: torch.Tensor) -> dict:
        device = x_t.device

        x_t = torch.cat((x_t, y_t), dim=1)

        batchsize = len(x_t)
        xy_tiled = torch.from_numpy(
            np.tile(self.xy, (batchsize, 1, 1, 1)).astype(np.float32)).to(device)

        z_tiled = torch.repeat_interleave(
            x_t, self.image_size*self.image_size, dim=0).view(batchsize, self.image_size, self.image_size, self.z_dim)

        z_and_xy = torch.cat((z_tiled, xy_tiled), dim=3)
        z_and_xy = z_and_xy.permute(0, 3, 2, 1)

        loc = self.loc(z_and_xy)/2.

        return Normal(loc, 0.001).rsample()


class Transition(nn.Module):
    """
      p(x_t | x_{t-1}, u_{t-1}; v_t) = N(x_t | x_{t-1} + ∆t·v_t, σ^2)
    """

    def __init__(self, delta_time: float):
        super().__init__()

        self.delta_time = delta_time

    def forward(self, x_tn1: torch.Tensor, v_t: torch.Tensor):

        x_t = x_tn1 + self.delta_time * v_t

        return Normal(x_t, 0.001).rsample()


class Velocity(nn.Module):
    """
        f(x_t | x_{t-1}, y_t) = f(x_t | x_{t-1}, y_t)
    """

    def __init__(self, delta_time, activate_func, device, use_data_efficiency):
        super().__init__()

        activation_func = getattr(nn, activate_func)
        self.delta_time = delta_time    
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
            self.A = torch.zeros((1, 3, 3)).to(device)    
            self.B = torch.zeros((1, 3, 3)).to(device)    
            self.C = torch.diag_embed(torch.ones(1, 3)).to(device)    
    
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

        # print(f"v_t: {v_t.detach().cpu().numpy()}", end="\r")

        return v_t


def main():
    encoder = Encoder(3, 10, 3, "ReLU")
    decoder = Decoder(3, 10, 3, "ReLU")
    velocity = Velocity(0.1, "ReLU", "cpu", True)
    transition = Transition(0.1)

    I_t = torch.randn(100, 5, 3, 64, 64)
    u_t = torch.randn(100, 5, 3)

    y = torch.randn(5, 10)
    x_tn1 = torch.zeros(5, 3)

    total_loss = 0.

    for t in range(1, 100-1):
        print(x_tn1)

        x_t = encoder.rsample(I_t[t], y)
        # print(x_t.shape)

        v_tn1 = x_t - x_tn1
        # print(v_t)

        v_t = velocity(x_tn1, v_tn1, u_t[t-1])
        # print(v_t)

        x_tp1 = transition(x_t, v_t)
        # print(x_t)

        I_tp1 = decoder(x_tp1, y)
        # print(I_hat_t.shape)

        rec_loss = nn.MSELoss()(I_tp1, I_t[t+1])
        kl_loss = nn.KLDivLoss(reduction="batchmean")(x_t, x_tn1)

        loss = rec_loss + kl_loss
        total_loss += loss

        x_tn1 = x_t

    print(loss)


if __name__ == "__main__":
    main()
