import os

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from pixyz.losses import Parameter, LogProb, KullbackLeibler as KL, Expectation as E, IterativeLoss
from pixyz.models import Model

from models.distributions import Encoder, Decoder, Transition, RotDecoder

# from distributions import Encoder, Decoder, Transition, RotDecoder
# import argparse
# import yaml
# import pprint
# from tqdm import tqdm

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
        self.encoder = torch.compile(Encoder(**encoder_param), mode="max-autotune").to(device)
        self.decoder = torch.compile(Decoder(**decoder_param), mode="max-autotune").to(device)
        self.rot_decoder = RotDecoder(input_dim=1, label_dim=1, output_dim=1).to(device)
        self.transition = Transition(**transition_param).to(device)

        #-------------------------#
        # Define loss functions   #
        #-------------------------#
        image_recon_loss = E(self.transition, LogProb(self.decoder))
        # rotation_recon_loss = E(self.transition, LogProb(self.rot_decoder))
        # recon_loss = (image_recon_loss + rotation_recon_loss).mean()
        recon_loss = (image_recon_loss).mean()

        kl_loss = KL(self.encoder, self.transition).mean()

        step_loss = - recon_loss + kl_loss

        loss = IterativeLoss(
                step_loss,
                max_iter=100,
                series_var=["I_t", "y_t", "R_t", "v_t"],
                update_value={"x_t_p": "x_tn1_p"}
        )

        super().__init__(loss=loss, distributions=[self.encoder, self.decoder, self.transition, self.rot_decoder], clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)

        self.delta_time = delta_time
        self.device = device


def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/train/point_mass.yml",
                        help='config path ex. config/sample/train/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)
        pprint.pprint(cfg)

    model = ConditionalNewtonianVAE(**cfg['model'])

    
    for i in tqdm(range(300)):
        for i in tqdm(range(1000)):
            I_t = torch.ones(100, 1, 3, 64, 64) * torch.arange(100).reshape(-1, 1, 1, 1, 1) + (torch.rand(100, 1, 3, 64, 64)-0.5)/500.
            I_t = I_t.cuda()/100.

            u = torch.ones(100, 1, 3) + (torch.rand(100, 1, 3)/10.-0.5)
            u[0] = torch.zeros(1, 1, 3)
            u = u.cuda()
            y = torch.zeros(100, 1, 2).cuda()
            R = torch.zeros(100, 1, 1).cuda()

            T, B, C = u.shape

            x_p_t = torch.zeros(B, C).to("cuda")

            loss, output_dict = model.train({'v_t': u, 'y_t': y, 'x_tn1_p': x_p_t, 'I_t': I_t, 'R_t': R}, return_dict=True)
            # print(output_dict["AnalyticalKullbackLeibler"])
            # loss = model.train({"I": I, "u": u, "y": y, "R": R})
            # print(loss)

        x = model.encoder.sample_mean({"I_t": I_t[0], "y_t": y[0]})
        print(x.cpu().detach().numpy().tolist())

if __name__ == "__main__":
    main()
