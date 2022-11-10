import numpy as np

import torch
from torch.utils.data import Dataset
    

class ExperienceReplay():
  def __init__(self, batch_size, sequence_size, action_size, device):
    self.device = device
    self.batch_size = batch_size
    self.sequence_size = sequence_size
    self.action_size = action_size

    self.colors = np.empty((batch_size, sequence_size, 3, 64, 64), dtype=np.float32)
    self.actions = np.empty((batch_size, sequence_size, action_size), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    # Tracks how much experience has been used in total
    self.steps, self.episodes = 0, 0

  def append(self, color, action, batch):
    self.colors[batch] = color
    self.actions[batch] = action

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n):
    idx = np.random.randint(0, self.batch_size, n)

    colors = torch.from_numpy(self.colors[idx].transpose(1, 0, 2, 3, 4)).to(self.device)
    actions = torch.from_numpy(self.actions[idx]).permute(1, 0, 2).to(self.device)

    return colors, actions

  def reset(self):
    self.colors = np.empty((self.size, 3, 64, 64), dtype=np.float32)
    self.actions = np.empty((self.size, self.action_size), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    # Tracks how much experience has been used in total
    self.steps, self.episodes = 0, 0
