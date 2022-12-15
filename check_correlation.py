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
from utils import memory
from environments import load


def main():
    # ================#
    # Load yaml file #
    # ================#
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument(
        '--config', type=str, help='config path ex. config/sample/check_correlation/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)
        pprint.pprint(cfg)

    env = load(**cfg["environment"])

    # ================#
    # Define model   #
    # ================#
    model = NewtonianVAE(**cfg["model"])
    model.load(**cfg["weight"])

    observation_position = []
    latent_position = []

    for i in range(5000):
        time_step = env.reset()

        video = env.physics.render(64, 64, camera_id=0)
        I_t = torch.tensor(video.transpose(2, 0, 1)[np.newaxis, :, :, :]/255.0).to(cfg["device"])

        x_q_t = model.encoder.sample_mean({"I_t": I_t})

        latent_position.append(x_q_t.to("cpu").detach().numpy()[0])
        observation_position.append(time_step.observation["position"][np.newaxis, :])

    all_observation_position = np.stack(observation_position)
    print(all_observation_position.shape)

    all_latent_position = np.stack(latent_position)
    print(all_latent_position.shape)

    value = np.corrcoef(
        all_observation_position[:, 0], all_latent_position[:, 0])
    print(value[0, 1])
    value = np.corrcoef(
        all_observation_position[:, 1], all_latent_position[:, 1])
    print(value[0, 1])

    for idx in range(len(all_latent_position)):
        color = list(colorsys.hsv_to_rgb(
            0.33*all_observation_position[idx, 0]/2., 0.33*all_observation_position[idx, 1], 0.5))
        color[2] = 0.
        plt.scatter(all_latent_position[idx, 0],
                    all_latent_position[idx, 1], color=color, s=2)
    plt.show()

    plt.scatter(all_observation_position[:, 0],
                all_latent_position[:, 0], s=2.)
    plt.show()

    plt.scatter(all_observation_position[:, 1],
                all_latent_position[:, 1], s=2.)
    plt.show()


if __name__ == "__main__":
    main()
