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

        self.colors_top = np.empty(
            (episode_size, sequence_size, 3, 64, 64), dtype=np.float32)
        self.colors_side = np.empty(
            (episode_size, sequence_size, 3, 64, 64), dtype=np.float32)
        self.colors_hand = np.empty(
            (episode_size, sequence_size, 3, 64, 64), dtype=np.float32)
        self.actions = np.empty(
            (episode_size, sequence_size, action_size), dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        colors_top = self.colors_top[index]
        colors_side = self.colors_side[index]
        colors_hand = self.colors_hand[index]
        actions = torch.from_numpy(self.actions[index])

        return _images_to_observation(colors_top, self.bit_depth), _images_to_observation(colors_side, self.bit_depth), _images_to_observation(colors_hand, self.bit_depth), actions

    def append(self, color_top, color_side, color_hand, action, batch):
        self.colors_top[batch] = postprocess_observation(color_top, self.bit_depth)
        self.colors_side[batch] = postprocess_observation(color_side, self.bit_depth)
        self.colors_hand[batch] = postprocess_observation(color_hand, self.bit_depth)
        self.actions[batch] = action

    def reset(self):
        self.colors_top = np.empty(
            (self.episode_size, 3, 64, 64), dtype=np.float32)
        self.colors_side = np.empty(
            (self.episode_size, 3, 64, 64), dtype=np.float32)
        self.colors_hand = np.empty(
            (self.episode_size, 3, 64, 64), dtype=np.float32)
        self.actions = np.empty(
            (self.episode_size, self.sequence_size, self.action_size), dtype=np.float32)

    def save(self, path, filename):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        np.savez(f"{path}/{filename}", **
                {"colors_top": self.colors_top, "colors_side": self.colors_side, "colors_hand": self.colors_hand, "actions": self.actions})

    def load(self, path, filename):
        """
        point-mass → colors_..., actions
        unity → I_..., action
        """
        with np.load(f"{path}/{filename}", allow_pickle=True) as data:
            self.colors_top = data["colors_top"][0:self.episode_size]
            self.colors_side = data["colors_side"][0:self.episode_size]
            self.colors_hand = data["colors_hand"][0:self.episode_size]
            self.actions = data["actions"][0:self.episode_size]
            # print(data["I_top"].shape)
            # self.colors_top = data["I_top"][0:self.episode_size]
            # self.colors_side = data["I_side"][0:self.episode_size]
            # self.colors_hand = data["I_hand"][0:self.episode_size]
            # self.actions = data["action"][0:self.episode_size]

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
