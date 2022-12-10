import os
import numpy as np

import torch
from torch.utils.data import Dataset


class ExperienceReplay():
    def __init__(self, episode_size, sequence_size, action_size, device):
        self.device = device
        self.episode_size = episode_size
        self.sequence_size = sequence_size
        self.action_size = action_size

        self.colors = np.empty(
            (episode_size, sequence_size, 3, 64, 64), dtype=np.float32)
        self.actions = np.empty(
            (episode_size, sequence_size, action_size), dtype=np.float32)
        self.positions = np.empty(
            (episode_size, sequence_size, action_size), dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        colors = torch.from_numpy(self.colors[index]).to(self.device)
        actions = torch.from_numpy(self.actions[index]).to(self.device)
        # positions = torch.from_numpy(self.positions[index]).to(self.device)

        return colors, actions# , positions

    def append(self, color, action, position, batch):
        self.colors[batch] = color
        self.actions[batch] = action
        # self.positions[batch] = position

    def reset(self):
        self.colors = np.empty(
            (self.episode_size, 3, 64, 64), dtype=np.float32)
        self.actions = np.empty(
            (self.episode_size, self.sequence_size, self.action_size), dtype=np.float32)
        # self.positions = np.empty(
        #     (self.episode_size, self.sequence_size, self.action_size), dtype=np.float32)

    def save(self, path, filename):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        np.savez(f"{path}/{filename}", **
                {"colors": self.colors, "actions": self.actions, "positions": self.positions})

    def load(self, path, filename):
        with np.load(f"{path}/{filename}", allow_pickle=True) as data:
            self.colors = data["colors"][0:self.episode_size]
            self.actions = data["actions"][0:self.episode_size]
            # self.positions = data["positions"]
