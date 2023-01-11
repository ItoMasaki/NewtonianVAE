#!/usr/bin/env python3
import pprint
import argparse
import yaml
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import memory, visualize
from utils.env import postprocess_observation
from environments import load, ControlSuiteEnv


def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/train/point_mass.yml",
                        help='config path e.g. config/sample/collect_dataset/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as _file:
        config = yaml.safe_load(_file)
        pprint.pprint(config)

    env = ControlSuiteEnv(**config["environment"])

    for mode in config["dataset"].keys():
        episode_size = config["dataset"][mode]["episode_size"]
        sequence_size = config["dataset"][mode]["sequence_size"]
        save_path = config["dataset"][mode]["data"]["path"]
        save_filename = config["dataset"][mode]["data"]["filename"]

        print(f"############## CONFIG PARAMS [{mode}] ##############")
        print(f"  max_episode : {episode_size}")
        print(f" max_sequence : {sequence_size}")
        print(f"    save_path : {save_path}")
        print(f"save_filename : {save_filename}")
        print(f"####################################################")

        save_memory = memory.ExperienceReplay(
            **config["dataset"][mode]["memory"])

        for episode in tqdm(range(episode_size)):
            time_step = env.reset()

            images_top = []
            images_side = []
            images_hand = []
            actions = []
            positions = []
            action = 0.

            for _ in range(sequence_size):
                action += np.random.uniform(-0.05, 0.05, 3)
                action[2] = abs(action[2])
                action = np.clip(action, -1, 1)

                if np.any(np.abs(action) >= 1.):
                    print("Warn : Over 1.")

                obs1, obs2, obs3, state, reward, done = env.step(torch.from_numpy(action))

                images_top.append(obs1.permute(2, 0, 1)[
                    np.newaxis, :, :, :])
                images_side.append(obs2.permute(2, 0, 1)[
                    np.newaxis, :, :, :])
                images_hand.append(obs3.permute(2, 0, 1)[
                    np.newaxis, :, :, :])
                actions.append(action[np.newaxis, :])

            save_memory.append(np.concatenate(images_top),
                               np.concatenate(images_side),
                               np.concatenate(images_hand),
                               np.concatenate(actions), episode)

        print()
        save_memory.save(save_path, save_filename)

        viz_top = visualize.Visualization()
        viz_side = visualize.Visualization()
        viz_hand = visualize.Visualization()

        for idx in range(len(images_top)):
            viz_top.append(postprocess_observation(images_top[idx][0].permute(1, 2, 0).numpy(), 8), postprocess_observation(images_top[idx][0].permute(1, 2, 0).numpy(), 8))

        for idx in range(len(images_side)):
           viz_side.append(postprocess_observation(images_side[idx][0].permute(1, 2, 0).numpy(), 8), postprocess_observation(images_side[idx][0].permute(1, 2, 0).numpy(), 8))

        for idx in range(len(images_hand)):
            viz_hand.append(postprocess_observation(images_hand[idx][0].permute(1, 2, 0).numpy(), 8), postprocess_observation(images_hand[idx][0].permute(1, 2, 0).numpy(), 8))
        viz_top.encode(save_path + "/videos/top", f"{mode}.mp4")
        viz_side.encode(save_path + "/videos/side", f"{mode}.mp4")
        viz_hand.encode(save_path + "/videos/hand", f"{mode}.mp4")



if __name__ == "__main__":
    main()
