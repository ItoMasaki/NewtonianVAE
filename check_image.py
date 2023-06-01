#!/usr/bin/env python3
import pprint
import argparse
import colorsys
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import animation

from models import ConditionalNewtonianVAE
from utils import memory, env
from environments import load, ControlSuiteEnv

import cv2


def main():

    max_number = 9

    #================#
    # Load yaml file #
    #================#
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument(
        '--config', type=str, default='config/sample/train/point_mass.yml', help='config path ex. config/sample/check_correlation/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)
        pprint.pprint(cfg)


    #================#
    # Define model   #
    #================#
    model = ConditionalNewtonianVAE(**cfg["model"])
    model.load(**cfg["weight"])

    observation_position = []
    latent_position = []

    fig, (ax1, ax2) = plt.subplots(1, 2)

    _x_yello = []
    _y_yello = []

    _x_brown = []
    _y_brown = []

    _x_black = []
    _y_black = []

    frames = []

    length = 0.35

    x_lim = [-length, length]
    y_lim = [-length, length]

    # 色範囲の指定
    lower_yello = np.array([207, 150, 65], dtype=np.uint8)
    upper_yello = np.array([228, 199, 130], dtype=np.uint8)

    lower_brown = np.array([75, 65, 58], dtype=np.uint8)
    upper_brown = np.array([96, 79, 80], dtype=np.uint8)


    for number in range(max_number):
        frames = []

        label = torch.eye(max_number)[number].cuda().unsqueeze(0)

        for i in np.linspace(*x_lim, 40):
            for j in np.linspace(*y_lim, 40):
                print(i, j)

                x_q_t = torch.Tensor([[i, j]])
                _image = model.decoder.sample_mean({"x_t": x_q_t.cuda(), "y_t": label})
                image = env.postprocess_observation(_image.cpu().detach().numpy(), 8).squeeze().transpose(1, 2, 0)

                # 色範囲内のピクセルを判定
                mask_yello = cv2.inRange(image, lower_yello, upper_yello)
                mask_brown = cv2.inRange(image, lower_brown, upper_brown)

                # mask内に1が存在するかを判定
                if (cv2.countNonZero(mask_yello) > 0) and not (cv2.countNonZero(mask_brown) > 0):
                    _x_yello.append(i)
                    _y_yello.append(j)
                elif not (cv2.countNonZero(mask_yello) > 0) and (cv2.countNonZero(mask_brown) > 0):
                    _x_brown.append(i)
                    _y_brown.append(j)
                elif not (cv2.countNonZero(mask_yello) > 0) and not (cv2.countNonZero(mask_brown) > 0):
                    _x_black.append(i)
                    _y_black.append(j)

                art1 = ax1.imshow(image)

                ax2.set_aspect('equal', adjustable='datalim')
                ax2.set_xlim(*x_lim)
                ax2.set_ylim(*y_lim)
                art2 = ax2.scatter(_x_yello, _y_yello, c="yellow")
                art3 = ax2.scatter(_x_brown, _y_brown, c="brown")
                art4 = ax2.scatter(_x_black, _y_black, c="black")

                frames.append([art1, art2, art3, art4])


        ani = animation.ArtistAnimation(fig, frames, interval=25)
        ani.save(f'animation.{number}.mp4', writer="ffmpeg")

if __name__ == "__main__":
    main()
