import os

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from pixyz.losses import KullbackLeibler as KL
from pixyz.losses import Expectation as E
from pixyz.losses import LogProb
from pixyz.losses import IterativeLoss
from pixyz.models import Model

from components.distributions import Encoder, Decoder, Transition, Velocity


class NewtonianVAE(Model):
  def __init__(self, optimizer: optim.Optimizer=optim.Adam, optimizer_params: dict={}, clip_grad_norm: bool=False, clip_grad_value: bool=False, delta_time: float=0.1, device: str="cuda"):
    #-------------------------#
    # Define models           #
    #-------------------------#
    self.encoder = Encoder(output_func_name="Tanh").to(device)
    self.decoder = Decoder().to(device)
    self.transition = Transition(delta_time).to(device)
    self.velocity = Velocity(delta_time).to(device)

    #-------------------------#
    # Define loss functions   #
    #-------------------------#
    self.recon_loss = E(self.transition, LogProb(self.decoder))
    self.kl_loss = KL(self.transition, self.encoder)
    self.step_loss = (self.kl_loss - self.recon_loss).mean()

    self.distributions = nn.ModuleList([self.encoder, self.decoder, self.transition, self.velocity])

    # set params and optim
    params = self.distributions.parameters()
    self.optimizer = optimizer(params, **optimizer_params)

    self.clip_norm = clip_grad_norm
    self.clip_value = clip_grad_value
    self.delta_time = delta_time
    self.device = device

  def calculate_loss(self, input_var_dict: dict={}):
    I = input_var_dict["I"]
    u = input_var_dict["u"]

    T, B, C = u.shape

    v = torch.zeros((B, 2)).to(self.device)
    x = self.encoder.sample({"I": I[0]}, reparam=True)["x"]

    total_loss = 0.

    for step in range(0, T-1):
      v = self.velocity(v_prev=v, x_prev=x, u=u[step])["v"]
      step_loss, variables = self.step_loss({"I": I[step+1], "x_prev": x, "v": v})

      x = variables["x"]

      total_loss += step_loss

    return total_loss/T
      
  def train(self, train_x_dict={}):
    self.distributions.train()

    self.optimizer.zero_grad()
    loss = self.calculate_loss(train_x_dict)

    # backprop
    loss.backward()

    if self.clip_norm:
        clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
    if self.clip_value:
        clip_grad_value_(self.distributions.parameters(), self.clip_value)

    # update params
    self.optimizer.step()

    return loss.item()

  def test(self, test_x_dict={}):
    self.distributions.eval()

    with torch.no_grad():
        loss = self.calculate_loss(test_x_dict)

    return loss.item()

  def infer(self, I: torch.Tensor, u: torch.Tensor, x_prev: torch.Tensor, v_prev: torch.Tensor):
    self.distributions.eval()

    with torch.no_grad():
      x_t = self.encoder.sample_mean({"I": I})
      I_t = self.decoder.sample_mean({"x": x_t})

      v_t = (x_t - x_prev)/self.delta_time

      x_next = self.transition.sample_mean({"x_prev":x_t, "v": v_t})
      I_next = self.decoder.sample_mean({"x": x_next})

    return I_t, x_t, v_t, I_next, x_next

  def save(self, path, filename):
    try:
      os.makedirs(path)
    except FileExistsError:
      pass

    torch.save({
      'distributions': self.distributions.state_dict(),
    }, f"{path}/{filename}")

  def init_params(self):

    for m in self.distributions.modules():

      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
              nn.init.constant_(m.bias, 0)

      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
