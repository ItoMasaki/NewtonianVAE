#!/usr/bin/env python3

from components.models import NewtonianVAE
from components.memory import ExperienceReplay

from dm_control import suite

import torch

import numpy as np

from matplotlib import pyplot as plt

from tensorboardX import SummaryWriter


batch_size = 10
max_episode = 10
max_sequence = 100
epoch_max = 100000
train_path = "datasets/sample/train.npz"
test_path = "datasets/sample/test.npz"

device = "cpu"
if torch.cuda.is_available():
  device = "cuda"

fig = plt.figure()



Train_Replay = ExperienceReplay(max_episode, max_sequence, 2, device)
Train_Replay.load(train_path)

Test_Replay = ExperienceReplay(1, max_sequence, 2, device)
Test_Replay.load(test_path)


writer = SummaryWriter(comment="NewtonianVAE")


model = NewtonianVAE()

for epoch in range(epoch_max):
  I, u = Train_Replay.sample(batch_size)

  train_loss = model.train({"I": I, "u": u})
  print(f"Epoch : {epoch} train_loss : {train_loss}", end="\r")
  writer.add_scalar('train_loss', train_loss, epoch)

  I, u = Test_Replay.sample(1)
  test_loss = model.test({"I": I, "u": u})
  writer.add_scalar('test_loss', test_loss, epoch)

  if epoch%100 == 0 and epoch != 0:

    v_t = torch.zeros((1, 2)).to(device)
    x_t = model.encoder.sample({"I": I[0]})["x"]

    all_positions = []

    for step in range(1, max_sequence):

      I_t, x_t, v_t, I_next, x_next = model.infer(I[step], u[step-1], x_t, v_t)

      all_positions.append(x_t.to("cpu").detach().numpy()[0].tolist())

      plt.clf()
      ax1 = fig.add_subplot(2, 2, 1)
      ax2 = fig.add_subplot(2, 2, 2)
      ax3 = fig.add_subplot(2, 2, 3)
      ax4 = fig.add_subplot(2, 2, 4)
      ax1.imshow(I.to("cpu").detach().numpy()[0].transpose(1, 2, 0))
      ax2.imshow(np.clip(I_t.to("cpu").detach().numpy()[0].transpose(1, 2, 0), 0., 1.))
      ax3.imshow(np.clip(I_next.to("cpu").detach().numpy()[0].transpose(1, 2, 0), 0., 1.))
      ax4.set_xlim(-1.1, 1.1)
      ax4.set_ylim(-1.1, 1.1)
      ax4.scatter(np.array(all_positions)[:, 0], np.array(all_positions)[:, 1], s=0.1)
      plt.pause(0.001)
