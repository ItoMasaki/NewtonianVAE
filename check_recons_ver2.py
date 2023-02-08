#!/usr/bin/env python3
import pprint
import os
import cv2
import argparse
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation

from models import NewtonianVAE
from utils import visualize, memory, env


def concat_images(
    img1, img2, img3, num_column=10, draw_line=True, line_color=(0, 0, 0), line_width=2, dummy_color=(0, 0, 0)
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


def plot_time_line(data, ax: plt.Axes, t: int, length=10):
    T, N = data.shape
    y_max = max(data.reshape(-1))
    y_min = min(data.reshape(-1))
    ax.set_ylim(y_min, y_max)
    if t < length:
        for i in range(N):
            ax.plot(np.arange(0, t + 1), data[: t + 1, i], marker="x")
        ax.set_xticks(np.arange(0, length + 1))
    else:
        for i in range(N):
            ax.plot(np.arange(t + 1 - length, t + 1), data[t + 1 - length : t + 1, i], marker="x")
        ax.set_xticks(np.arange(t + 1 - length, t + 1))


def plot_multi_bars(data, ax: plt.Axes, t: int, label=["x", "y", "z"]):
    T, N = data.shape
    max_delta = np.max(np.abs(data)) + 0.05
    x = np.arange(N).tolist()
    ax.bar(x, data[t], color="b", align="center")
    ax.set_xticks(x, label)
    ax.set_ylim(-max_delta, max_delta)
    # zero line
    ax.axhline(y=0, color="k", linestyle="-", linewidth=1)


def plot_obs_recons(obs, recons, n_frame=None, dpi=100, each_figsize=3, base_path="obs-recon"):
    n_frame = len(obs["top"]) if n_frame is None else n_frame

    w_graph = 2
    h_graph = 3
    fig, axes = plt.subplots(h_graph, w_graph, figsize=(each_figsize * w_graph, each_figsize * h_graph), dpi=dpi)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def plot(t):
        plt.cla()
        for i in range(h_graph):
            for j in range(w_graph):
                axes[i, j].cla()
                axes[i, j].axis("off")
        fig.suptitle("t={}".format(t))

        # plot top
        axes[0, 0].imshow(obs["top"][t])
        axes[0, 1].imshow(recons["top"][t])
        # plot side
        axes[1, 0].imshow(obs["side"][t])
        axes[1, 1].imshow(recons["side"][t])
        # plot hand
        axes[2, 0].imshow(obs["hand"][t])
        axes[2, 1].imshow(recons["hand"][t])

    anim = FuncAnimation(fig, plot, frames=n_frame, interval=100)
    save_path = "{}_obs-recon.mp4".format(base_path)
    print("save movie : {}".format(save_path))
    anim.save(save_path, writer="ffmpeg", dpi=dpi)
    plt.close()


def plot_obs_recons_lspos(obs, recons, n_frame=None, dpi=100, each_figsize=3, base_path="obs-recon"):
    n_frame = len(obs["top"]) if n_frame is None else n_frame

    w_graph = 3
    h_graph = 3
    fig, axes = plt.subplots(h_graph, w_graph, figsize=(each_figsize * w_graph, each_figsize * h_graph), dpi=dpi)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def plot(t):
        plt.cla()
        for i in range(h_graph):
            for j in range(w_graph):
                axes[i, j].cla()
                axes[i, j].axis("off")
        fig.suptitle("t={}".format(t))

        # plot top
        axes[0, 0].imshow(obs["top"][t])
        axes[0, 1].imshow(recons["top"][t])
        # plot side
        axes[1, 0].imshow(obs["side"][t])
        axes[1, 1].imshow(recons["side"][t])
        # plot hand
        axes[2, 0].imshow(obs["hand"][t])
        axes[2, 1].imshow(recons["hand"][t])
        # plot lspos
        axes[0, 2].axis("on")
        axes[1, 2].set_title("latent state")
        plot_time_line(recons["latent_position"], axes[0, 2], t, length=10)

    anim = FuncAnimation(fig, plot, frames=n_frame, interval=100)
    save_path = "{}_obs-recon-lspos.mp4".format(base_path)
    print("save movie : {}".format(save_path))
    anim.save(save_path, writer="ffmpeg", dpi=dpi)
    plt.close()


def plot_obs_recons_action_lspos(obs, recons, n_frame=None, dpi=100, each_figsize=3, base_path="obs-recon"):
    n_frame = len(obs["top"]) if n_frame is None else n_frame

    w_graph = 3
    h_graph = 3
    fig, axes = plt.subplots(h_graph, w_graph, figsize=(each_figsize * w_graph, each_figsize * h_graph), dpi=dpi)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def plot(t):
        plt.cla()
        for i in range(h_graph):
            for j in range(w_graph):
                axes[i, j].cla()
                axes[i, j].axis("off")
        fig.suptitle("t={}".format(t))

        # plot top
        axes[0, 0].imshow(obs["top"][t])
        axes[0, 1].imshow(recons["top"][t])
        # plot side
        axes[1, 0].imshow(obs["side"][t])
        axes[1, 1].imshow(recons["side"][t])
        # plot hand
        axes[2, 0].imshow(obs["hand"][t])
        axes[2, 1].imshow(recons["hand"][t])
        # plot action
        axes[0, 2].set_title("action")
        axes[0, 2].axis("on")
        plot_multi_bars(obs["action"], axes[0, 2], t, ["x", "y", "z"])
        # plot lspos
        axes[1, 2].set_title("latent position")
        axes[1, 2].axis("on")
        plot_time_line(recons["latent_position"], axes[1, 2], t, length=10)

    anim = FuncAnimation(fig, plot, frames=n_frame, interval=100)
    save_path = "{}_obs-recon-action-lspos.mp4".format(base_path)
    print("save movie : {}".format(save_path))
    anim.save(save_path, writer="ffmpeg", dpi=dpi)
    plt.close()


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
        # pprint.pprint(cfg)

    # ======================#
    # plot hyperparameters  #
    # ======================#
    output_dir = "output"
    weight_file_list = [
        "300.weight",
    ]
    mode = "train"
    dataset_path = cfg["dataset"][mode]["data"]["path"]
    filename = cfg["dataset"][mode]["data"]["filename"]
    epi_idx_list = [0, 1]
    num_img_divide = 10
    num_column = 10

    # ==============#
    # Load dataset  #
    # ==============#
    if os.path.exists(dataset_path) == False:
        raise FileNotFoundError("Dataset not found")
    D = memory.ExperienceReplay(**cfg["dataset"][mode]["memory"])
    D.load(dataset_path, filename)

    # ================#
    # Define model   #
    # ================#
    model = NewtonianVAE(**cfg["model"])

    for weight_filename in weight_file_list:
        model.load(cfg["weight"]["path"], weight_filename)
        for epi_idx in epi_idx_list:
            I_top, I_side, I_hand, u = D[epi_idx]
            epi_length = cfg["dataset"][mode]["sequence_size"] - 1
            if epi_length != I_top.shape[0] - 1:
                print("Warning: episode length is not matched.")
            if epi_length > I_top.shape[0] - 1:
                epi_length = I_top.shape[0] - 1

            obs = dict(
                top=env.postprocess_observation(
                    I_hand[:epi_length].to("cpu", non_blocking=True).detach().numpy().transpose(0, 2, 3, 1),
                    cfg["bit_depth"],
                ),
                side=env.postprocess_observation(
                    I_side[:epi_length].to("cpu", non_blocking=True).detach().numpy().transpose(0, 2, 3, 1),
                    cfg["bit_depth"],
                ),
                hand=env.postprocess_observation(
                    I_hand[:epi_length].to("cpu", non_blocking=True).detach().numpy().transpose(0, 2, 3, 1),
                    cfg["bit_depth"],
                ),
                action=u[:epi_length].to("cpu", non_blocking=True).detach().numpy(),
            )
            recons = dict(
                top=np.empty((epi_length, 64, 64, 3), dtype=np.uint8),
                side=np.empty((epi_length, 64, 64, 3), dtype=np.uint8),
                hand=np.empty((epi_length, 64, 64, 3), dtype=np.uint8),
                latent_position=np.empty((epi_length, 3), dtype=np.float32),
            )

            for step in range(epi_length):
                I_top_t, I_side_t, I_hand_t, I_top_tp1, I_side_tp1, I_hand_tp1, x_q_t, x_p_tp1 = model.estimate(
                    I_top.to(cfg["device"], non_blocking=True).unsqueeze(1)[step + 1],
                    I_side.to(cfg["device"], non_blocking=True).unsqueeze(1)[step + 1],
                    I_hand.to(cfg["device"], non_blocking=True).unsqueeze(1)[step + 1],
                    I_top.to(cfg["device"], non_blocking=True).unsqueeze(1)[step],
                    I_side.to(cfg["device"], non_blocking=True).unsqueeze(1)[step],
                    I_hand.to(cfg["device"], non_blocking=True).unsqueeze(1)[step],
                    u.to(cfg["device"], non_blocking=True).unsqueeze(1)[step + 1],
                )

                recons["top"][step] = env.postprocess_observation(
                    I_top_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[0].transpose(1, 2, 0),
                    cfg["bit_depth"],
                )
                recons["side"][step] = env.postprocess_observation(
                    I_side_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[0].transpose(1, 2, 0),
                    cfg["bit_depth"],
                )
                recons["hand"][step] = env.postprocess_observation(
                    I_hand_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[0].transpose(1, 2, 0),
                    cfg["bit_depth"],
                )
                recons["latent_position"][step] = x_q_t.to("cpu", non_blocking=True).detach().numpy()[0]

            base_name = "weight-{}_mode-{}_epi-{}".format(os.path.splitext(weight_filename)[0], mode, epi_idx)
            base_path = os.path.join(output_dir, base_name)

            # save images
            img_idx_list = np.linspace(0, epi_length - 1, num_img_divide, dtype=np.uint32).tolist()

            whole_img = concat_images(
                recons["top"][img_idx_list],
                recons["side"][img_idx_list],
                recons["hand"][img_idx_list],
                num_column,
                line_width=5,
            )
            cv2.imwrite("{}_recon.png".format(base_path), whole_img[:, :, ::-1])
            whole_img = concat_images(
                obs["top"][img_idx_list], obs["side"][img_idx_list], obs["hand"][img_idx_list], num_column, line_width=5
            )
            cv2.imwrite("{}_obs.png".format(base_path), whole_img[:, :, ::-1])

            # save movie
            plot_obs_recons(obs=obs, recons=recons, base_path=base_path)
            plot_obs_recons_lspos(obs=obs, recons=recons, base_path=base_path)
            plot_obs_recons_action_lspos(obs=obs, recons=recons, base_path=base_path)

            del obs, recons


if __name__ == "__main__":
    main()
