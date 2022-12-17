#!/usr/bin/env python3
import pprint
import argparse
import yaml
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import memory
from environments import load, ControlSuiteEnv


def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/collect_dataset/point_mass.yml",
                        help='config path e.g. config/sample/collect_dataset/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as _file:
        config = yaml.safe_load(_file)
        pprint.pprint(config)

    env = ControlSuiteEnv(**config["environment"])

    episode_size = config["dataset"]["test"]["episode_size"]
    sequence_size = config["dataset"]["test"]["sequence_size"]
    save_path = config["dataset"]["test"]["save_path"]
    save_filename = config["dataset"]["test"]["save_filename"]

    action = 0.

    for episode in tqdm(range(episode_size)):
        time_step = env.reset()

        for _ in range(sequence_size):
            env.render()
            action += np.random.uniform(-0.05, 0.05, 2)
            action = np.clip(action, -1., 1.)

            observation, reward, done = env.step(torch.from_numpy(action))


if __name__ == "__main__":
    main()
