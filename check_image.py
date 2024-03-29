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

from models import NewtonianVAE
from utils import memory, env
from environments import load, ControlSuiteEnv

import cv2


def main():
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
    model = NewtonianVAE(**cfg["model"])
    model.load(**cfg["weight"])

    observation_position = []
    latent_position = []

    fig = plt.figure(figsize = (10, 5))

    _x_yello = []
    _y_yello = []

    _x_brown = []
    _y_brown = []

    _x_black = []
    _y_black = []

    frames = []

    x_lim = [-2, 2]
    y_lim = [-2, 2]

    # 色範囲の指定
    lower_yello = np.array([207, 150, 65], dtype=np.uint8)
    upper_yello = np.array([228, 199, 130], dtype=np.uint8)

    lower_brown = np.array([75, 65, 58], dtype=np.uint8)
    upper_brown = np.array([96, 79, 80], dtype=np.uint8)

    frames = []

    for i in np.linspace(*x_lim, 40):
        for j in np.linspace(*y_lim, 40):

            x_q_t = torch.Tensor([[i, j]])
            _image = model.decoder.sample_mean({"x_t": x_q_t.cuda()})
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

            plt.clf()

            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            art1 = ax1.imshow(image)
            frames.append([art1])

            ax2.set_xlim(*x_lim)
            ax2.set_ylim(*y_lim)
            ax2.set_aspect('equal')

            art2 = ax2.scatter(_x_yello, _y_yello, c="yellow")
            art3 = ax2.scatter(_x_brown, _y_brown, c="brown")
            art4 = ax2.scatter(_x_black, _y_black, c="black")
            frames.append([art2])
            frames.append([art3])
            frames.append([art4])

            plt.pause(0.01)

    plt.show()

    # アニメーションの作成
    ani = animation.ArtistAnimation(fig, frames, interval=100)
    ani.save('check_image.mp4', writer="ffmpeg")

if __name__ == "__main__":
    main()
