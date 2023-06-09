#!/usr/bin/env python3
import os
import pprint
import argparse
import yaml
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from models import ConditionalNewtonianVAE
from utils import env
from environments import load, ControlSuiteEnv



def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/point_mass.yml",
                        help='config path e.g. config/sample/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as _file:
        cfg = yaml.safe_load(_file)
        pprint.pprint(cfg)

    
    save_root_path = cfg["weight"]["path"].replace("weights", "operability")
    os.makedirs(save_root_path, exist_ok=True)

    #================#
    # Define model   #
    #================#
    model = ConditionalNewtonianVAE(**cfg["model"])
    model.load(**cfg["weight"])

    for episode in tqdm(range(90)):
        frames = []
        #==================#
        # Get target image #
        #==================#
        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()

        number = 0 # time_step[2]
        label = torch.eye(1)[number].cuda().unsqueeze(0)

        target_observation, state, reward, done = _env.step(torch.zeros(1, 3))
        target_x_q_t = model.encoder.sample_mean({"I_t": target_observation.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
            "y_t": label})

        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()
        
        number = 0 # time_step[2]
        label = torch.eye(1)[number].cuda().unsqueeze(0)

        action = torch.zeros(1, 3)
        
        fig, ((axis1, axis2), (axis3, axis4)) = plt.subplots(2, 2, tight_layout=True)

        for _ in range(1000):
            #===================#
            # Get current image #
            #===================#
            observation, state, reward, done = _env.step(action.cpu())

            error_from_origin = state.observation["position"][:3]

            x_q_t = model.encoder.sample_mean({
                "I_t": observation.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
                "y_t": label})
            reconstructed_image = model.decoder.sample_mean({"x_t": x_q_t, "y_t": label})

            #============#
            # Get action #
            #============#
            action = x_q_t.detach()

            action = - action * 0.01
            print(action)

            # action = torch.flip(action, dims=[1]) * 0.1
            # action = -torch.flip(action, dims=[1]) * 0.1
            # action[0, 1] = -action[0, 1]
            # action[0, 0] = -action[0, 0]

            axis1.set_title("Controlling")
            art1 = axis1.imshow(env.postprocess_observation(observation.detach().numpy(), 8))

            axis2.set_title("Reconstructed")
            art2 = axis2.imshow(env.postprocess_observation(reconstructed_image[0].permute(1, 2, 0).cpu().detach().numpy(), 8))

            axis3.set_title("Action")
            axis3.set_ylim(-0.5, 0.5)
            _action = action[0].cpu().detach().numpy().tolist()
            bar1, bar2, bar3 = axis3.bar(["X", "Y", "R"], _action, color=["black", "black", "black"])

            axis4.set_title("Error from origin")
            axis4.set_ylim(-3.0, 3.0)
            bar4, bar5, bar6 = axis4.bar(["X", "Y", "R"], error_from_origin, color=["black", "black", "black"])

            frames.append([art1, art2, bar1, bar2, bar3, bar4, bar5, bar6])

        ani = animation.ArtistAnimation(fig, frames, interval=10)
        ani.save(f"{save_root_path}/output.{episode}.mp4", writer="ffmpeg")


if __name__ == "__main__":
    main()
