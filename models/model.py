import os

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.distributions import Normal

from pixyz.losses import Parameter, LogProb, KullbackLeibler as KL, Expectation as E
from pixyz.models import Model
from pixyz import distributions as dist

from models.distributions import Encoder1, Encoder2, Encoder3, Decoder1, Decoder2, Decoder3, Transition, Velocity

torch.backends.cudnn.benchmark = True


class NewtonianVAE(Model):
    def __init__(self,
                 encoder_param: dict = {},
                 decoder_param: dict = {},
                 transition_param: dict = {},
                 velocity_param: dict = {},
                 optimizer: str = "Adam",
                 optimizer_params: dict = {},
                 clip_grad_norm: bool = False,
                 clip_grad_value: bool = False,
                 delta_time: float = 0.5,
                 device: str = "cuda",
                 use_amp: bool = False):

        # -------------------------#
        # Define models           #
        # -------------------------#
        self.encoder1 = Encoder1(**encoder_param).to(device)
        self.encoder2 = Encoder2(**encoder_param).to(device)
        self.encoder3 = Encoder3(**encoder_param).to(device)
        self.decoder1 = Decoder1(**decoder_param).to(device)
        self.decoder2 = Decoder2(**decoder_param).to(device)
        self.decoder3 = Decoder3(**decoder_param).to(device)
        self.transition = Transition(**transition_param).to(device)
        self.velocity = Velocity(**velocity_param).to(device)

        # -------------------------#
        # Define hyperparams      #
        # -------------------------#
        # beta = Parameter("beta")/
        # self.phi generates 3-dim samples
        phi_dim = 3

        # ----------------------------#
        # MoPoE                       #
        # ----------------------------#
        # poe
        self.x_top_t = dist.ProductOfNormal([self.encoder1])
        self.x_side_t = dist.ProductOfNormal([self.encoder2])
        self.x_hand_t = dist.ProductOfNormal([self.encoder3])
        self.x_topside_t = dist.ProductOfNormal([self.encoder1, self.encoder2])
        self.x_sidehand_t = dist.ProductOfNormal([self.encoder2, self.encoder3])
        self.x_handtop_t = dist.ProductOfNormal([self.encoder3, self.encoder1])
        self.x_topsidehand_t = dist.ProductOfNormal([self.encoder1, self.encoder2, self.encoder3])
        self.phi = dist.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=['x_t'], features_shape=[phi_dim], name='p_{1}')

        # moe
        self.x_moe = dist.MixtureOfNormal([self.x_top_t, self.x_side_t, self.x_hand_t, self.x_topside_t, self.x_sidehand_t, self.x_handtop_t, self.x_topsidehand_t])
        # print("x_q_t=", self.x_moe)

        # -------------------------#
        # Define loss functions   #
        # -------------------------#
        # Reconstruction losses
        recon_loss_top = E(self.transition, LogProb(self.decoder1))
        recon_loss_side = E(self.transition, LogProb(self.decoder2))
        recon_loss_hand = E(self.transition, LogProb(self.decoder3))
        # KL divergence
        kl_loss = KL(self.x_top_t, self.transition)+KL(self.x_side_t, self.transition)+KL(self.x_hand_t, self.transition)+KL(self.x_topside_t, self.transition)+KL(self.x_sidehand_t, self.transition)+KL(self.x_handtop_t, self.transition)+KL(self.x_topsidehand_t, self.transition)+KL(self.phi, self.transition)
        # Step loss
        self.step_loss_top = (kl_loss - recon_loss_top).mean()
        self.step_loss_side = (kl_loss - recon_loss_side).mean()
        self.step_loss_hand = (kl_loss - recon_loss_hand).mean()
        # print("self.recon_los_top=", self.step_loss_top)
        # print("self.recon_los_top=", self.step_loss_top)
        # print("self.recon_los_top=", self.step_loss_top)
        self.step_loss = self.step_loss_top + self.step_loss_side + self.step_loss_hand

        self.distributions = nn.ModuleList(
            [self.x_top_t, self.x_side_t, self.x_hand_t, self.x_topside_t, self.x_sidehand_t, self.x_handtop_t, self.x_topsidehand_t,self.x_moe, self.decoder1, self.decoder2, self.decoder3, self.transition, self.velocity])
            # self.phi,

        # -------------------------#
        # Set params and optim     #
        # -------------------------#
        params = self.distributions.parameters()
        self.optimizer = getattr(optim, optimizer)(params, **optimizer_params)

        # -------------------------------------------------#
        # Set for AMP                                      #
        # Whather to use Automatic Mixture Precision [AMP] #
        # -------------------------------------------------#
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.clip_norm = clip_grad_norm
        self.clip_value = clip_grad_value
        self.delta_time = delta_time
        self.device = device

    def calculate_loss(self, input_var_dict: dict = {}):

        I_top = input_var_dict["I_top_t"]
        I_side = input_var_dict["I_side_t"]
        I_hand = input_var_dict["I_hand_t"]
        u = input_var_dict["u"]
        # beta = input_var_dict["beta"]

        total_loss = 0.

        T, B, C = u.shape

        # x^q_{t-1} ~ p(x^q_{t-1} | I_{t-1})
        x_q_tn1 = self.x_moe.sample({"I_top_t": I_top[0], "I_side_t": I_side[0], "I_hand_t": I_hand[0]}, reparam=True)["x_t"]

        # print("I_top.shape = ", I_top.shape)

        for step in range(1, T-1):
            # x^q_{t} ~ p(x^q_{t} | I_{t})
            x_q_t = self.x_moe.sample({"I_top_t": I_top[step], "I_side_t": I_side[step], "I_hand_t": I_hand[step]}, reparam=True)["x_t"]

            # v_t = (x^q_{t} - x^q_{t-1})/dt
            v_t = (x_q_t - x_q_tn1)/self.delta_time

            # v_{t+1} = v_{t} + dt (A*x_{t} + B*v_{t} + C*u_{t})
            v_tp1 = self.velocity(x_tn1=x_q_t, v_tn1=v_t, u_tn1=u[step])["v_t"]

            # KL[p(x^p_{t+1} | x^q_{t}, u_{t}; v_{t+1}) || q(x^q_{t+1} | I_{t+1})] - E_p(x^p_{t+1} | x^q_{t}, u_{t}; v_{t+1})[log p(I_{t+1} | x^p_{t+1})]
            step_loss, variables = self.step_loss({'x_tn1': x_q_t, 'v_t': v_tp1, 'I_top_t': I_top[step+1], 'I_side_t': I_side[step+1], 'I_hand_t': I_hand[step+1]})

            total_loss += step_loss

            x_q_tn1 = x_q_t

        return total_loss/T

    def train(self, train_x_dict={}):

        self.distributions.train()

        with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP
            loss = self.calculate_loss(train_x_dict)

        self.optimizer.zero_grad(set_to_none=True)
        # self.optimizer.zero_grad()

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
            with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP
                loss = self.calculate_loss(test_x_dict)

        return loss.item()

    def estimate(self, I_top_t: torch.Tensor, I_side_t: torch.Tensor, I_hand_t: torch.Tensor, I_top_tn1: torch.Tensor, I_side_tn1: torch.Tensor, I_hand_tn1: torch.Tensor, u_t: torch.Tensor):
        self.distributions.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP

                # x^q_{t-1} ~ p(x^q_{t-1) | I_{t-1))
                x_q_tn1 = self.x_moe.sample_mean({"I_top_t": I_top_tn1, "I_side_t": I_side_tn1, "I_hand_t": I_hand_tn1})

                # x^q_t ~ p(x^q_t | I_t)
                x_q_t = self.x_moe.sample_mean({"I_top_t": I_top_t, "I_side_t": I_side_t, "I_hand_t": I_hand_t})

                # p(I_t | x_t)
                I_top_t = self.decoder1.sample_mean({"x_t": x_q_t})
                I_side_t = self.decoder2.sample_mean({"x_t": x_q_t})
                I_hand_t = self.decoder3.sample_mean({"x_t": x_q_t})

                # v_t = (x^q_t - x^q_{t-1})/dt
                v_t = (x_q_t - x_q_tn1)/self.delta_time

                # v_{t+1} = v_{t} + dt (A*x_{t} + B*v_{t} + C*u_{t})
                v_tp1 = self.velocity(x_tn1=x_q_t, v_tn1=v_t, u_tn1=u_t)["v_t"]

                # p(x_p_{t+1} | x_q_{t}, v_{t+1})
                x_p_tp1 = self.transition.sample_mean(
                    {"x_tn1": x_q_t, "v_t": v_tp1})

                # p(I_{t+1} | x_{t+1})
                I_top_tp1 = self.decoder1.sample_mean({"x_t": x_p_tp1})
                I_side_tp1 = self.decoder2.sample_mean({"x_t": x_p_tp1})
                I_hand_tp1 = self.decoder3.sample_mean({"x_t": x_p_tp1})

        return I_top_t, I_side_t, I_hand_t, I_top_tp1, I_side_tp1, I_hand_tp1, x_q_t, x_p_tp1

    def save(self, path, filename):
        os.makedirs(path, exist_ok=True)

        torch.save({
            'distributions': self.distributions.to("cpu").state_dict(),
        }, f"{path}/{filename}")

        self.distributions.to(self.device)

    def save_ckpt(self, path, filename, epoch, loss):
        os.makedirs(path, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.distributions.to("cpu").state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, f"{path}/{filename}")

        self.distributions.to(self.device)

    def load(self, path, filename):
        self.distributions.load_state_dict(torch.load(
            f"{path}/{filename}", map_location=torch.device('cpu'))['distributions'])

    def load_ckpt(self, path, filename):
        self.distributions.load_state_dict(torch.load(
            f"{path}/{filename}", map_location=torch.device('cpu'))['distributions']['model_state_dict'])

        self.optimizer.load_state_dict(torch.load(
            f"{path}/{filename}", map_location=torch.device('cpu'))['distributions']['optimizer_state_dict'])
