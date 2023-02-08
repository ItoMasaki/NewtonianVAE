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
import cv2

from models import NewtonianVAE
from utils import env
from environments import load, ControlSuiteEnv


fig, (axis1, axis2, axis3) = plt.subplots(1, 3, tight_layout=True)


def concat_images(
    img1, img2, img3, num_column=10, draw_line=True, line_color=(0, 0, 0), line_width=5, dummy_color=(0, 0, 0)
):
    T, H, W, C = img1.shape
    if draw_line:
        img_list = [img1, img2, img3]
        whole_img = []
        for i in range(0, T, num_column):
            row_img = []
            for j in range(3):
                row_img_list = []
                for k in range(num_column):
                    if i + k >= T:
                        dummy_img = np.full((H, W, C), dummy_color, dtype=np.uint8)
                        row_img_list.append(dummy_img)
                    else:
                        row_img_list.append(img_list[j][i + k])
                    if k != num_column - 1:
                        # vertical line
                        row_img_list.append(np.full((H, line_width, C), line_color, dtype=np.uint8))
                row_img.append(np.concatenate(row_img_list, axis=1))
                if j != 2:
                    # horizontal line
                    row_img.append(np.full((line_width, row_img[0].shape[1], C), line_color, dtype=np.uint8))
            if i + num_column < T:
                # horizontal line 2
                row_img.append(np.full((line_width, row_img[0].shape[1], C), line_color, dtype=np.uint8))
            row_img = np.concatenate(row_img, axis=0)
            whole_img.append(row_img)
        whole_img = np.concatenate(whole_img, axis=0)
    else:
        whole_img = []
        for i in range(0, T, num_column):
            row_img = np.zeros((H * 3, W * num_column, C), dtype=np.uint8)
            # row_img = np.full((H * 3, W * num_column, C), 255, dtype=np.uint8)
            if i + num_column > T:
                num_column = T - i
            row_img[H * 0 : H * 1, : W * num_column, :] = np.concatenate(img1[i : i + num_column], axis=1)
            row_img[H * 1 : H * 2, : W * num_column, :] = np.concatenate(img2[i : i + num_column], axis=1)
            row_img[H * 2 : H * 3, : W * num_column, :] = np.concatenate(img3[i : i + num_column], axis=1)
            whole_img.append(row_img)
        whole_img = np.concatenate(whole_img, axis=0)
    return whole_img


def main():
    parser = argparse.ArgumentParser(description="Collection dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sample/point_mass.yml",
        help="config path e.g. config/sample/point_mass.yml",
    )
    args = parser.parse_args()

    with open(args.config) as _file:
        cfg = yaml.safe_load(_file)
        pprint.pprint(cfg)

    # ================#
    # Define model   #
    # ================#
    model = NewtonianVAE(**cfg["model"])
    model.load(**cfg["weight"])

    # ======================#
    # plot hyperparameters  #
    # ======================#
    num_episode = 10
    horizon_length = 400
    num_img_divide = 10
    num_column = 10

    for episode in tqdm(range(num_episode)):
        frames = []
        # ==================#
        # Get target image #
        # ==================#
        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()
        target_obs1, target_obs2, target_obs3, state, reward, done = _env.step(torch.zeros(1, 3))
        # print("traget_obs1 =", target_obs1)
        target_x_q_t = model.x_moe.sample_mean(
            {
                "I_top_t": target_obs1.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
                "I_side_t": target_obs2.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
                "I_hand_t": target_obs3.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
            }
        )
        # target_x_q_t = torch.zeros(3)
        # print("target_x_q_t =", target_x_q_t)
        # target_x_q_t = model.encoder.sample_mean({"I_t": target_obs1.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"])})

        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()
        action = torch.zeros(1, 3)

        images = dict(
            top=np.empty((horizon_length, 64, 64, 3), dtype=np.uint8),
            side=np.empty((horizon_length, 64, 64, 3), dtype=np.uint8),
            hand=np.empty((horizon_length, 64, 64, 3), dtype=np.uint8),
        )

        for t in range(horizon_length):
            # ===================#
            # Get current image #
            # ===================#
            obs1, obs2, obs3, state, reward, done = _env.step(action.cpu())
            x_q_t = model.x_moe.sample_mean(
                {
                    "I_top_t": obs1.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
                    "I_side_t": obs2.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
                    "I_hand_t": obs3.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
                }
            )
            # if _ == 0:
            #     print("x_q_t =", x_q_t)
            # x_q_t = model.encoder.sample_mean({"I_t": observation.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"])})

            # ============#
            # Get action #
            # ============#
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

            images["top"][t] = env.postprocess_observation(obs1.detach().numpy(), 8)
            images["side"][t] = env.postprocess_observation(obs2.detach().numpy(), 8)
            images["hand"][t] = env.postprocess_observation(obs3.detach().numpy(), 8)

            if t == (horizon_length-1):
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

        img_idx_list = np.linspace(0, horizon_length-1, num_img_divide, dtype=np.uint32).tolist()
        # print(f"img idx:{img_idx_list}")
        whole_img = np.concatenate(
            [
                np.concatenate(images["top"][img_idx_list], axis=1),
                np.concatenate(images["side"][img_idx_list], axis=1),
                np.concatenate(images["hand"][img_idx_list], axis=1),
            ],
            axis=0,
        )
        whole_img2 = concat_images(
            images["top"][img_idx_list],
            images["side"][img_idx_list],
            images["hand"][img_idx_list],
            num_column=num_column,
        )

        # cv2.imwrite(f"output.{episode}.png", whole_img[:, :, ::-1])
        cv2.imwrite(f"output.{episode}_v2.png", whole_img2[:, :, ::-1])
        # cv2.imwrite(f"output.{episode}.png", whole_img)


if __name__ == "__main__":
    main()
