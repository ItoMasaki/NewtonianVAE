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

            labels = np.array([time_step[2]]*sequence_size).reshape(-1, 1)
            images = []
            actions = []
            positions = []
            action = np.zeros(3)

            first_rot = 0.
            not_first_flag = True
            position = np.zeros(3)

            transition_limit = 0.25
            rotation_limit = 1.

            for _ in range(sequence_size):
                action[0] = action[0] + np.random.uniform(-0.005, 0.005, 1)
                action[1] = action[1] + np.random.uniform(-0.005, 0.005, 1)
                action[2] = action[2] + np.random.uniform(-0.5, 0.5, 1)
                action[:1] = np.clip(action[:1], -transition_limit, transition_limit)
                # print(action)

                # Rotation
                if position[2] < -3.1:
                    action[2] = np.clip(action[2], 0, rotation_limit)
                elif position[2] > 3.1:
                    action[2] = np.clip(action[2], -rotation_limit, 0)
                else:
                    action[2] = np.clip(action[2], -rotation_limit, rotation_limit)


                observation, state, reward, done = env.step(torch.from_numpy(action))
                # Render
                env.render()

                images.append(observation.permute(2, 0, 1)[
                    np.newaxis, :, :, :])
                actions.append(action[np.newaxis, :].copy())

                if not_first_flag:
                    first_rot = state.observation["position"][2]
                    not_first_flag = False

                position = state.observation["position"][:3]
                position[2] -= first_rot

                print(f"position : {position} action : {action}", end="\r")

                positions.append(position[np.newaxis, :])

            save_memory.append(np.concatenate(images),
                               np.concatenate(actions), np.concatenate(positions), labels, episode)

        print()
        save_memory.save(save_path, save_filename)


if __name__ == "__main__":
    main()
