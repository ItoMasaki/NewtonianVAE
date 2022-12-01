#!/usr/bin/env python3
import argparse
import colorsys
import numpy as np
import yaml
import torch
from matplotlib import pyplot as plt

from models import NewtonianVAE
from utils import memory
from environments import load


def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument(
        '--config', type=str, help='config path ex. config/sample/check_correlation/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    env_name = cfg["env"]
    episodes = cfg["episodes"]
    weight_path = cfg["weight_path"]
    weight_filename = cfg["weight_filename"]

    model = NewtonianVAE(**cfg["model"])
    model.load(f"{weight_path}", f"{weight_filename}")

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))

    if env_name == "reacher_nvae":
        env = load(domain_name="reacher_nvae", task_name="hard",
                   task_kwargs={"whole_range": False})

    elif env_name == "reacher":
        env = load(domain_name="reacher", task_name="hard")

    elif env_name == "point_mass":
        env = load(domain_name="point_mass", task_name="easy")

    else:
        raise NotImplementedError(f"{config['env']}")

    observation_position = []
    latent_position = []

    for episode in range(1000):
        time_step = env.reset()

        action = np.random.uniform(-1, 1, 2)

        video = env.physics.render(64, 64, camera_id=0)
        I_t = torch.tensor(video.copy().transpose(2, 0, 1)[
                             np.newaxis, :, :, :]/255).to(torch.float32).to(cfg["device"])

        x_q_t = model.encoder.sample_mean({"I_t": I_t})

        observation_position.append(time_step.observation["position"])
        latent_position.append(x_q_t.to("cpu").detach().numpy()[0])

        # for _ in range(100):
        #     time_step = env.step(action)
        #     u = torch.tensor(action[np.newaxis, :]).to(
        #         torch.float32).to(cfg["device"])

        #     video = env.physics.render(64, 64, camera_id=0)
        #     I_t = torch.tensor(video.copy().transpose(2, 0, 1)[
        #                        np.newaxis, :, :, :]/255).to(torch.float32).to(cfg["device"])

        #     rec_I_t, I_tp1, x_q_t, x_p_tp1 = model.estimate(I_t, I_tn1, u)

        #     observation_position.append(time_step.observation["position"])
        #     latent_position.append(x_q_t.to("cpu").detach().numpy()[0])
        #     plt.imshow(rec_I_t.to("cpu").detach().to(torch.float32).numpy()[0].transpose(1, 2, 0))
        #     # ax1.imshow(I_t.to("cpu").detach().to(torch.float32).numpy()[0].transpose(1, 2, 0))
        #     # ax2.imshow(rec_I_t.to("cpu").detach().to(torch.float32).numpy()[0].transpose(1, 2, 0))
        #     # ax3.imshow(rec_I_tp1.to("cpu").detach().to(torch.float32).numpy()[0].transpose(1, 2, 0))
        #     # ax4.scatter(x_q_t.to("cpu").detach().numpy()[:, 0], x_q_t.to("cpu").detach().numpy()[:, 1], s=1.)
        #     plt.pause(0.01)

        #     plt.clf()

        #     I_tn1 = I_t

    print("Visualization")

    all_observation_position = np.stack(observation_position)
    print(all_observation_position.shape)
    all_latent_position = np.stack(latent_position)
    print(all_latent_position.shape)

    value = np.corrcoef(
        all_observation_position[:, 0], all_latent_position[:, 0])
    print(value[0, 1])
    value = np.corrcoef(
        all_observation_position[:, 0], all_latent_position[:, 1])
    print(value[0, 1])
    value = np.corrcoef(
        all_observation_position[:, 1], all_latent_position[:, 0])
    print(value[0, 1])
    value = np.corrcoef(
        all_observation_position[:, 1], all_latent_position[:, 1])
    print(value[0, 1])

    X = all_observation_position[:, 0] / \
        np.abs(all_observation_position[:, 0]).max()
    Y = all_observation_position[:, 1] / \
        np.abs(all_observation_position[:, 1]).max()

    for idx in range(len(all_latent_position)):

        # print(X[idx], Y[idx], colorsys.hls_to_rgb(X[idx], .5, Y[idx]))

        color = list(colorsys.hls_to_rgb(X[idx]/2., .5, Y[idx]))
        color[2] = 0.

        plt.scatter(all_latent_position[idx, 0],
                    all_latent_position[idx, 1], color=color, s=2)

    plt.show()

    plt.scatter(all_observation_position[:, 0], all_latent_position[:, 0], s=2.)
    plt.show()

    plt.scatter(all_observation_position[:, 0], all_latent_position[:, 1], s=2.)
    plt.show()

    plt.scatter(all_observation_position[:, 1], all_latent_position[:, 0], s=2.)
    plt.show()

    plt.scatter(all_observation_position[:, 1], all_latent_position[:, 1], s=2.)
    plt.show()


if __name__ == "__main__":
    main()
