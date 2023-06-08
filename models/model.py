import os

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from pixyz.losses import Parameter, LogProb, KullbackLeibler as KL, Expectation as E, IterativeLoss
from pixyz.models import Model

from models.distributions import Encoder, Decoder, Transition, Velocity, RotDecoder

from timm.scheduler import CosineLRScheduler

# from distributions import Encoder, Decoder, Transition, Velocity, LabelEncoder
# import argparse
# import yaml
# import pprint

torch.backends.cudnn.benchmark = True


class ConditionalNewtonianVAE(Model):
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

        #-------------------------#
        # Define models           #
        #-------------------------#
        self.encoder = torch.compile(Encoder(**encoder_param)).to(device)
        self.decoder = torch.compile(Decoder(**decoder_param)).to(device)
        self.rot_decoder = torch.compile(RotDecoder(input_dim=1, label_dim=1, output_dim=1, activate_func="ReLU")).to(device)
        self.transition = Transition(**transition_param).to(device)
        self.velocity = Velocity(**velocity_param).to(device)

        #-------------------------#
        # Define hyperparams      #
        #-------------------------#
        beta = Parameter("beta")

        #-------------------------#
        # Define loss functions   #
        #-------------------------#
        rot_recon_loss = E(self.transition, LogProb(self.rot_decoder))
        recon_loss = E(self.transition, LogProb(self.decoder))
        kl_loss = KL(self.encoder, self.transition, analytical=True)
        self.loss_cls = (beta*kl_loss - recon_loss - rot_recon_loss).mean()

        self.distributions = nn.ModuleList(
            [self.encoder, self.decoder, self.transition, self.velocity, self.rot_decoder])

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

        I = input_var_dict["I"]
        u = input_var_dict["u"]
        y = input_var_dict["y"]
        R = input_var_dict["R"]
        beta = input_var_dict["beta"]

        total_loss = 0.

        encoded_pos = []

        T, B, C = u.shape

        # x^q_{t-1} ~ p(x^q_{t-1} | I_{t-1})
        # x_q_tn1 = self.encoder.sample({"I_t": I[0], "y_t": y[0]}, reparam=True)["x_t"]
        x_q_tn1 = torch.zeros(B, C).to(self.device)

        encoded_pos.append(x_q_tn1)

        for step in range(1, T-1):

            # x^q_{t} ~ p(x^q_{t} | I_{t})
            x_q_t = self.encoder.sample({"I_t": I[step], "y_t": y[step]}, reparam=True)["x_t"]

            encoded_pos.append(x_q_t)

            # v_t = (x^q_{t} - x^q_{t-1})/dt
            v_t = (x_q_t - x_q_tn1)/self.delta_time

            # v_{t+1} = v_{t} + dt (A*x_{t} + B*v_{t} + C*u_{t})
            v_tp1 = self.velocity(x_tn1=x_q_t, v_tn1=v_t, u_tn1=u[step])["v_t"]

            # KL[p(x^p_{t+1} | x^q_{t}, u_{t}; v_{t+1}) || q(x^q_{t+1} | I_{t+1})] - E_p(x^p_{t+1} | x^q_{t}, u_{t}; v_{t+1})[log p(I_{t+1} | x^p_{t+1})]
            step_loss, variables = self.loss_cls({'x_tn1': x_q_t, 'v_t': v_tp1, 'I_t': I[step+1], 'y_t': y[step+1], 'beta': beta, "R_t": R[step+1]})

            total_loss += step_loss

            x_q_tn1 = x_q_t

        return total_loss/T, torch.stack(encoded_pos, dim=0)

    def train(self, train_x_dict={}):

        self.distributions.train()

        with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP
            loss, pos = self.calculate_loss(train_x_dict)

        self.optimizer.zero_grad(set_to_none=True)

        # backward
        self.scaler.scale(loss).backward()

        if self.clip_norm:
            clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
        if self.clip_value:
            clip_grad_value_(self.distributions.parameters(), self.clip_value)

        # update params
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), pos

    def test(self, test_x_dict={}):

        self.distributions.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP
                loss, pos = self.calculate_loss(test_x_dict)

        return loss.item(), pos

    def estimate(self, I_t: torch.Tensor, I_tn1: torch.Tensor, u_t: torch.Tensor, y: torch.Tensor):
        self.distributions.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP

                # x^q_{t-1} ~ p(x^q_{t-1) | I_{t-1))
                x_q_tn1 = self.encoder.sample_mean({"I_t": I_tn1, "y_t": y})

                # x^q_t ~ p(x^q_t | I_t)
                x_q_t = self.encoder.sample_mean({"I_t": I_t, "y_t": y})

                # p(I_t | x_t)
                I_t = self.decoder.sample_mean({"x_t": x_q_t, "y_t": y})

                # v_t = (x^q_t - x^q_{t-1})/dt
                v_t = (x_q_t - x_q_tn1)/self.delta_time

                # v_{t+1} = v_{t} + dt (A*x_{t} + B*v_{t} + C*u_{t})
                v_tp1 = self.velocity(x_tn1=x_q_t, v_tn1=v_t, u_tn1=u_t)["v_t"]

                # p(x_p_{t+1} | x_q_{t}, v_{t+1})
                x_p_tp1 = self.transition.sample_mean(
                    {"x_tn1": x_q_t, "v_t": v_tp1})

                # p(I_{t+1} | x_{t+1})
                I_tp1 = self.decoder.sample_mean({"x_t": x_p_tp1, "y_t": y})

        return I_t, I_tp1, x_q_t, x_p_tp1

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

    @staticmethod
    def create_epochs_and_others(num_epochs=100, repeat=1, num_steps_per_epoch=10,):
        num_epochs = num_epochs
        num_epoch_repeat = num_epochs//repeat
        num_steps_per_epoch = num_steps_per_epoch
        return num_epochs, num_epoch_repeat, num_steps_per_epoch

def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/train/point_mass.yml",
                        help='config path ex. config/sample/train/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)
        pprint.pprint(cfg)

    model = ConditionalNewtonianVAE(**cfg['model'])
    print(model)

    I = torch.randn(300, 1, 3, 64, 64).cuda()
    u = torch.randn(300, 1, 2).cuda()
    y = torch.randn(300, 1, 2).cuda()

    loss = model.train({"I": I, "u": u, "y": y, "beta": 1.})
    print(loss)


if __name__ == "__main__":
    main()
