import os

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from pixyz.losses import KullbackLeibler as KL
from pixyz.losses import Expectation as E
from pixyz.losses import LogProb
from pixyz.losses import IterativeLoss
from pixyz.models import Model

from models.distributions import Encoder, Decoder, Transition, Velocity

torch.backends.cudnn.benchmark = True

class NewtonianVAE(Model):
  def __init__(self,
          optimizer: optim.Optimizer=optim.Adam,
          optimizer_params: dict={},
          clip_grad_norm: bool=True,
          clip_grad_value: bool=True,
          delta_time: float=0.5,
          device: str="cuda",
          use_amp: bool=True):
    #-------------------------#
    # Define models           #
    #-------------------------#
    self.encoder = Encoder().to(device)
    self.decoder = Decoder().to(device)
    self.transition = Transition(delta_time).to(device)
    self.velocity = Velocity(delta_time).to(device)

    #-------------------------#
    # Define loss functions   #
    #-------------------------#
    recon_loss = E(self.transition, LogProb(self.decoder))
    kl_loss = KL(self.transition, self.encoder)
    self.step_loss = (kl_loss - recon_loss).mean()

    self.distributions = nn.ModuleList([self.encoder, self.decoder, self.transition, self.velocity])

    #-------------------------#
    # set params and optim    #
    #-------------------------#
    params = self.distributions.parameters()
    self.optimizer = optimizer(params, lr=3e-4, **optimizer_params)

    #-------------------------#
    # set for AMP             #
    #-------------------------#
    self.use_amp = use_amp # Whather to use Automatic Mixture Precision [AMP]
    self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    self.clip_norm = clip_grad_norm
    self.clip_value = clip_grad_value
    self.delta_time = delta_time
    self.device = device

  def calculate_loss(self, input_var_dict: dict={}):

    I = input_var_dict["I"]
    u = input_var_dict["u"]

    total_loss = 0.

    T, B, C = u.shape

    # x^q_tn1 ~ p(x^q_tn1 | I_tn1)
    x_q_tn1 = self.encoder.sample({"I_t": I[0]}, reparam=True)["x_t"]

    for step in range(1, T-1):

      # x^q_t ~ p(x^q_t | I_t)
      x_q_t = self.encoder.sample({"I_t": I[step]}, reparam=True)["x_t"]

      # v_t = (x^q_t - x^q_tn1)/dt
      v_t = (x_q_t - x_q_tn1)/self.delta_time

      # v_{t+1} = v_t + dt (A*x_t + B*v_t + C*u_t)
      v_tp1 = self.velocity(x_tn1=x_q_t, v_tn1=v_t, u_tn1=u[step])["v_t"]

      # KL[q(x^q_{t+1} | I_{t+1}) || p(x^p_{t+1} | x^p_t, u_t; v_{t+1})] - E_p(x^p_{t+1} | x^p_t, u_t; v_{t+1})[log p(I_{t+1} | x^p_{t+1})]
      step_loss, variables = self.step_loss({'x_tn1': x_q_t, 'v_t': v_tp1, 'I_t': I[step+1]})

      total_loss += step_loss

      x_q_tn1 = x_q_t

    return total_loss/T
      
  def train(self, train_x_dict={}):

    self.distributions.train()

    self.optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.use_amp): # AMP
      loss = self.calculate_loss(train_x_dict)

    # backward
    self.scaler.scale(loss).backward()

    if self.clip_norm:
        clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
    if self.clip_value:
        clip_grad_value_(self.distributions.parameters(), self.clip_value)

    # update params
    self.scaler.step(self.optimizer)

    # update scaler
    self.scaler.update()

    return loss.item()

  def test(self, test_x_dict={}):

    self.distributions.eval()

    with torch.no_grad():
      with torch.cuda.amp.autocast(enabled=self.use_amp): # AMP
        loss = self.calculate_loss(test_x_dict)

    return loss.item()

  def estimate(self, I_t: torch.Tensor, I_tn1: torch.Tensor, u_t: torch.Tensor):
    self.distributions.eval()

    with torch.no_grad():
      with torch.cuda.amp.autocast(enabled=self.use_amp): # AMP

        # x^q_{t-1} ~ p(x^q_{t-1) | I_{t-1))
        x_q_tn1 = self.encoder.sample_mean({"I_t": I_tn1})

        # x^q_t ~ p(x^q_t | I_t)
        x_q_t = self.encoder.sample_mean({"I_t": I_t})

        # p(I_t | x_t)
        I_t = self.decoder.sample_mean({"x_t": x_q_t})

        # v_t = (x^q_t - x^q_{t-1})/dt
        v_t = (x_q_t - x_q_tn1)/self.delta_time

        # v_{t+1} = v_t + dt (A*x_t + B*v_t + C*u_t)
        v_tp1 = self.velocity(x_tn1=x_q_t, v_tn1=v_t, u_tn1=u_t)["v_t"]

        # p(x_p_{t+1} | x_p_t, v_{t+1})
        x_p_tp1 = self.transition.sample_mean({"x_tn1":x_q_t, "v_t": v_tp1})

        # p(I_{t+1} | x_{t+1})
        I_tp1 = self.decoder.sample_mean({"x_t": x_p_tp1})

    return I_t, I_tp1, x_q_t, x_p_tp1

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