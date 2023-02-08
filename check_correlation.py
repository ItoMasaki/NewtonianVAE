#!/usr/bin/env python3
import pprint
import argparse
import colorsys
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from models import NewtonianVAE
from utils import memory, env
from environments import load, ControlSuiteEnv


def main():
    # ================#
    # Load yaml file #
    # ================#
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument(
        '--config', type=str, default='config/sample/point_mass.yml', help='config path ex. config/sample/check_correlation/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)
        pprint.pprint(cfg)

    _env = ControlSuiteEnv(**cfg["environment"])

    # ================#
    # Define model   #
    # ================#
    model = NewtonianVAE(**cfg["model"])
    model.load(**cfg["weight"])

    for episode in range(10):
        observation_position = []
        latent_position = []

        for i in range(600):
            obs1, obs2, obs3, state = _env.reset()

            obs1, obs2, obs3, state, reward, done = _env.step(torch.zeros(3))

            x_q_t = model.x_moe.sample_mean({"I_top_t": obs1.permute(2, 0, 1)[np.newaxis, :, :, :], "I_side_t": obs2.permute(2, 0, 1)[np.newaxis, :, :, :], "I_hand_t": obs3.permute(2, 0, 1)[np.newaxis, :, :, :]})

            latent_position.append(x_q_t.to("cpu").detach().numpy()[0])
            observation_position.append(state.observation["position"])
            # print("state.observation =", state.observation["position"])

        all_observation_position = np.stack(observation_position)
        # print("all_observation_position.shape =", all_observation_position.shape)
        # print(all_observation_position[10])

        all_latent_position = np.stack(latent_position)
        # print("all_latent_position.shape =", all_latent_position.shape)
        # print(all_latent_position[1])

        value = np.corrcoef(
            all_observation_position[:, 0], all_latent_position[:, 2])
        print(f"corrcoef 0 = {round(value[0,1], 3)}")
        value = np.corrcoef(
            all_observation_position[:, 1], all_latent_position[:, 1])
        print(f"corrcoef 1 = {round(value[0,1], 3)}")
        value = np.corrcoef(
            all_observation_position[:, 2], all_latent_position[:, 0])
        print(f"corrcoef 2 = {round(value[0,1], 3)}")
        print("\n")

        for idx in range(len(all_latent_position)):
            color = list(colorsys.hsv_to_rgb((all_observation_position[idx, 0] + 0.1)*2.5, (all_observation_position[idx, 1] + 0.1)*5.0, 0.5))
            plt.scatter(all_latent_position[idx, 0],
                        all_latent_position[idx, 1], color=color, s=2)
        plt.xlabel("latent state 0")
        plt.ylabel("latent state 1")
        plt.savefig(f"latent_xy_episode{episode}.eps")
        plt.show()

        for idx in range(len(all_latent_position)):
            color = list(colorsys.hsv_to_rgb((all_observation_position[idx, 1] + 0.1)*2.5, (all_observation_position[idx, 2] + 0.1)*5.0, 0.5))
            plt.scatter(all_latent_position[idx, 1],
                        all_latent_position[idx, 2], color=color, s=2)
        plt.xlabel("latent state 1")
        plt.ylabel("latent state 2")
        plt.savefig(f"latent_yz_episode{episode}.eps")
        plt.show()

        for idx in range(len(all_latent_position)):
            color = list(colorsys.hsv_to_rgb((all_observation_position[idx, 0] + 0.1)*2.5, (all_observation_position[idx, 2] + 0.1)*5.0, 0.5))
            plt.scatter(all_latent_position[idx, 0],
                        all_latent_position[idx, 2], color=color, s=2)
        plt.xlabel("latent state 0")
        plt.ylabel("latent state 2")
        plt.savefig(f"latent_xz_episode{episode}.eps")
        plt.show()

        plt.scatter(all_observation_position[:, 0],
                    all_latent_position[:, 2], s=2.)
        plt.xlabel("ground truth x")
        plt.ylabel("latent state 2")
        plt.savefig(f"correlation_x_episode{episode}.eps")
        plt.show()

        plt.scatter(all_observation_position[:, 1],
                    all_latent_position[:, 1], s=2.)
        plt.xlabel("ground truth y")
        plt.ylabel("latent state 1")
        plt.savefig(f"correlation_y_episode{episode}.eps")
        plt.show()

        plt.scatter(all_observation_position[:, 2],
                    all_latent_position[:, 0], s=2.)
        plt.xlabel("ground truth z")
        plt.ylabel("latent state 0")
        plt.savefig(f"correlation_z_episode{episode}.eps")
        plt.show()

if __name__ == "__main__":
    main()
