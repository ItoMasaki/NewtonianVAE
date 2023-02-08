#!/usr/bin/env python3
import pprint
import argparse
import yaml
import numpy as np
import torch
import math
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from models import NewtonianVAE
from utils import env
from environments import load, ControlSuiteEnv


fig, (axis1, axis2, axis3) = plt.subplots(1, 3, tight_layout=True)

def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/point_mass.yml",
                        help='config path e.g. config/sample/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as _file:
        cfg = yaml.safe_load(_file)
        pprint.pprint(cfg)


    # ================#
    # Define model   #
    # ================#
    model = NewtonianVAE(**cfg["model"])
    model.load(**cfg["weight"])

    for episode in tqdm(range(10)):
        frames = []
        #==================#
        # Get target image #
        #==================#
        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()
        target_obs1, target_obs2, target_obs3, state, reward, done = _env.step(torch.zeros(1, 3))
        # print("traget_obs1 =", target_obs1)
        target_x_q_t = model.x_moe.sample_mean({"I_top_t": target_obs1.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]), "I_side_t": target_obs2.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]), "I_hand_t": target_obs3.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"])})
        # target_x_q_t = torch.zeros(3)
        # print("target_x_q_t =", target_x_q_t)
        # target_x_q_t = model.encoder.sample_mean({"I_t": target_obs1.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"])})

        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()
        action = torch.zeros(1, 3)
        for _ in range(400):
            #===================#
            # Get current image #
            #===================#
            obs1, obs2, obs3, state, reward, done = _env.step(action.cpu())
            x_q_t = model.x_moe.sample_mean({"I_top_t": obs1.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]), "I_side_t": obs2.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]), "I_hand_t": obs3.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"])})
            # if _ == 0:
            #     print("x_q_t =", x_q_t)
            # x_q_t = model.encoder.sample_mean({"I_t": observation.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"])})

            #============#
            # Get action #
            #============#
            action = target_x_q_t - x_q_t
            # print("action =", action)

            # art1 = axis1.imshow(env.postprocess_observation(target_obs2.detach().numpy(), 8))
            # art2 = axis2.imshow(env.postprocess_observation(obs2.detach().numpy(), 8))
            axis1.axis("off")
            art1 = axis1.imshow(env.postprocess_observation(obs1.detach().numpy(), 8))
            axis2.axis("off")
            art2 = axis2.imshow(env.postprocess_observation(obs2.detach().numpy(), 8))
            axis3.axis("off")
            art3 = axis3.imshow(env.postprocess_observation(obs3.detach().numpy(), 8))


            frames.append([art1, art2, art3])

            if _ == 399:
                # print(x_q_t[0][0], x_q_t[0][1], x_q_t[0][2])
                # print(target_x_q_t)
                # print(x_q_t.shape)
                # print("pow((target_x_q_t[0]-x_q_t[0]), 2) = ", pow((target_x_q_t[0][0]-x_q_t[0][0]), 2))
                # print("pow((target_x_q_t[1]-x_q_t[1]), 2) = ", pow((target_x_q_t[0][1]-x_q_t[0][1]), 2))
                # print("pow((target_x_q_t[1]-x_q_t[1]), 2) = ", pow((target_x_q_t[0][2]-x_q_t[0][2]), 2))
                position_x = state.observation["position"][0]
                position_y = state.observation["position"][1]
                position_z = state.observation["position"][2]
                print(f"episode{episode} = ", math.sqrt(pow((position_x-0), 2) + pow((position_y-0), 2) + pow((position_z-0), 2)))
                print(f"distance_x = {math.sqrt(pow((position_x-0), 2))}")
                print(f"distance_y = {math.sqrt(pow((position_y-0), 2))}")
                print(f"distance_z = {math.sqrt(pow((position_z-0), 2))}")
                print("\n")

        ani = animation.ArtistAnimation(fig, frames, interval=10)
        ani.save(f"output.{episode}.mp4", writer="ffmpeg")


if __name__ == "__main__":
    main()
