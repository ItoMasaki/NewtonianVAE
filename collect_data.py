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

import time


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

            labels = np.array([time_step[3]]*sequence_size).reshape(-1, 1)
            images = []
            depthes = []
            actions = []
            positions = []
            position = np.zeros(4)
            height = 0.
            action = np.zeros(4)

            first_rot = 0.
            not_first_flag = True

            for _ in range(sequence_size):
                action[0] += height * np.random.uniform(-0.01, 0.01, 1)
                action[1] += height * np.random.uniform(-0.01, 0.01, 1)
                action[2] += np.random.uniform(-0.01, 0.02, 1)
                action[3] += np.random.uniform(-0.5, 0.5, 1)

                # action[0] = 0
                # action[1] = 0
                # action[2] = 0
                # action[3] = 0

                height += action[2]/3.

                action[0] = np.clip(action[0], -0.9, 0.9)
                action[1] = np.clip(action[1], -0.9, 0.9)

                # Height
                if height < -0.:
                    action[2] = np.clip(action[2], 0.00, 2.00)
                else:
                    action[2] = np.clip(action[2], -0.20, 0.200)

                # Rotation
                if position[3] < -3.:
                    action[3] = np.clip(action[3], 0, 2)
                elif position[3] > 3.:
                    action[3] = np.clip(action[3], -2, 0)
                else:
                    action[3] = np.clip(action[3], -2, 2)
                # print(action)

                observation, depth, state, reward, done = env.step(torch.from_numpy(action))
                # print(state.observation["position"])
                env.render()

                images.append(observation.permute(2, 0, 1)[
                    np.newaxis, :, :, :])

                depthes.append(depth[np.newaxis, :, :, np.newaxis].transpose(0, 3, 1, 2))

                actions.append(action[np.newaxis, :])

                if not_first_flag:
                    first_pose = state.observation["position"][:4]
                    not_first_flag = False

                position = state.observation["position"][:4] - first_pose
                # print(position)

                positions.append(position[np.newaxis, :])

                # time.sleep(.1)

            # print(np.concatenate(positions).shape)
            # print(np.concatenate(depthes).shape)

            save_memory.append(
                    np.concatenate(images),
                    np.concatenate(depthes),
                    np.concatenate(actions),
                    np.concatenate(positions), labels, episode)

        print()
        save_memory.save(save_path, save_filename)


if __name__ == "__main__":
    main()
