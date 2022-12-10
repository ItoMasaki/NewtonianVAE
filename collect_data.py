#!/usr/bin/env python3
import argparse
import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import memory
from environments import load


parser = argparse.ArgumentParser(description='Collection dataset')
parser.add_argument('--config', type=str,
                    help='config path ex. config/sample/collect_dataset/point_mass.yml')
args = parser.parse_args()

with open(args.config) as file:
    config = yaml.safe_load(file)


for mode in config.keys():
    env_name = config["train"]["env"]

    if env_name == "reacher_nvae":
        if mode == "test":
            env = load(domain_name="reacher_nvae", task_name="hard",
                       task_kwargs={"whole_range": False})
        else:
            env = load(domain_name="reacher_nvae", task_name="hard",
                       task_kwargs={"whole_range": True})

    elif env_name == "reacher":
        env = load(domain_name="reacher", task_name="hard")

    elif env_name == "point_mass":
        env = load(domain_name="point_mass", task_name="easy")

    else:
        raise NotImplementedError(f"{config['env']}")

    max_episode = config[mode]["max_episode"]
    max_sequence = config[mode]["max_sequence"]
    save_path = config[mode]["save_path"]
    save_filename = config[mode]["save_filename"]

    print(f"############## CONFIG PARAMS [{mode}] ##############")
    print(f"  max_episode : {max_episode}")
    print(f" max_sequence : {max_sequence}")
    print(f"    save_path : {save_path}")
    print(f"save_filename : {save_filename}")
    print(f"####################################################")

    save_memory = memory.ExperienceReplay(max_episode, max_sequence, 2, "cpu")

    print("Collect data")
    for episode in tqdm(range(max_episode)):
        time_step = env.reset()

        actions = []
        observations = []

        # direction = np.random.uniform(-1, 1, 2)
        action = np.random.uniform(-1, 1, 2)

        for _ in range(max_sequence):
            # if mode == "train":
            #   action = np.random.uniform(-1, 1, 2)
            # else:
            #   action = np.array([np.random.normal(direction[0], 0.1), np.random.normal(direction[1]/2., 0.1)])
            time_step = env.step(action)
            video = env.physics.render(64, 64, camera_id=0)

            actions.append(action[np.newaxis, :])
            observations.append(video.transpose(2, 0, 1)[
                                np.newaxis, :, :, :]/255.0)

        save_memory.append(np.concatenate(observations),
                           np.concatenate(actions), episode)

    print()
    save_memory.save(save_path, save_filename)
