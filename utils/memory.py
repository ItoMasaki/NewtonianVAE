import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from utils.env import postprocess_observation, _images_to_observation


class ExperienceReplay():
    def __init__(self, episode_size, sequence_size, action_size, bit_depth, device):
        self.device = device
        self.episode_size = episode_size
        self.sequence_size = sequence_size
        self.action_size = action_size
        self.bit_depth = bit_depth

        self.colors = np.empty(
            (episode_size, sequence_size, 3, 64, 64), dtype=np.float32)
        self.depthes = np.empty(
            (episode_size, sequence_size, 1, 64, 64), dtype=np.float32)
        self.actions = np.empty(
            (episode_size, sequence_size, action_size), dtype=np.float32)
        self.positions = np.empty(
            (episode_size, sequence_size, action_size), dtype=np.float32)
        self.labels = np.empty(
            (episode_size, sequence_size, 1), dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        colors = self.colors[index]
        depthes = self.depthes[index] * 100.
        actions = torch.from_numpy(self.actions[index]).float()
        positions = torch.from_numpy(self.positions[index]).float()
        labels = torch.from_numpy(self.labels[index]).float()

        return _images_to_observation(colors, self.bit_depth), _images_to_observation(depthes, self.bit_depth), actions, positions, labels

    def append(self, color, depth, action, position, label, batch):
        self.colors[batch] = postprocess_observation(color, self.bit_depth)
        self.depthes[batch] = depth
        self.actions[batch] = action
        self.positions[batch] = position
        self.labels[batch] = label

    def reset(self):
        self.colors = np.empty(
            (self.episode_size, 3, 64, 64), dtype=np.float32)
        self.depthes = np.empty(
            (self.episode_size, 1, 64, 64), dtype=np.float32)
        self.actions = np.empty(
            (self.episode_size, self.sequence_size, self.action_size), dtype=np.float32)
        self.positions = np.empty(
            (episode_size, sequence_size, action_size), dtype=np.float32)
        self.labels = np.empty(
            (episode_size, sequence_size, 1), dtype=np.float32)

    def save(self, path, filename):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        np.savez(f"{path}/{filename}", **
                {"colors": self.colors, "depthes":self.depthes, "actions": self.actions, "positions": self.positions, "labels": self.labels})

    def load(self, path, filename):
        with np.load(f"{path}/{filename}", allow_pickle=True) as data:
            self.colors = data["colors"][0:self.episode_size]
            self.depthes = data["depthes"][0:self.episode_size]
            self.actions = data["actions"][0:self.episode_size]
            self.positions = data["positions"][0:self.episode_size]
            self.labels = data["labels"][0:self.episode_size]

def make_loader(cfg, mode):
    #==========================#
    # Define experiment replay #
    #==========================#
    replay = ExperienceReplay(**cfg["dataset"][mode]["memory"])
     
    #==============#
    # Load dataset #
    #==============#
    replay.load(**cfg["dataset"][mode]["data"])
     
    #====================#
    # Define data loader #
    #====================#
    loader = DataLoader(replay, **cfg["dataset"][mode]["loader"])

    return loader
