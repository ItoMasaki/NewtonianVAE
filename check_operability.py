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


fig, (axis1, axis2) = plt.subplots(1, 2, tight_layout=True)

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

        number = time_step[2]
        label = torch.eye(10)[number].cuda().unsqueeze(0)

        target_observation, state, reward, done = _env.step(torch.zeros(1, 2))
        target_x_q_t = model.encoder.sample_mean({"I_t": target_observation.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
            "y_t": label})

        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()
        
        number = time_step[2]
        label = torch.eye(10)[number].cuda().unsqueeze(0)

        action = torch.zeros(1, 2)
        for _ in range(200):
            #===================#
            # Get current image #
            #===================#
            observation, state, reward, done = _env.step(action.cpu())

            x_q_t = model.encoder.sample_mean({
                "I_t": observation.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
                "y_t": label})
            reconstructed_image = model.decoder.sample_mean({"x_t": x_q_t, "y_t": label})

            #============#
            # Get action #
            #============#
            action = (target_x_q_t - x_q_t).detach()

            # action = -torch.flip(action, dims=[1])
            # action[0, 1] = -action[0, 1]

            art1 = axis1.imshow(env.postprocess_observation(target_observation.detach().numpy(), 8))
            art2 = axis2.imshow(env.postprocess_observation(observation.detach().numpy(), 8))

            frames.append([art1, art2])

        ani = animation.ArtistAnimation(fig, frames, interval=10)
        ani.save(f"{save_root_path}/output.{episode}.mp4", writer="ffmpeg")


if __name__ == "__main__":
    main()
