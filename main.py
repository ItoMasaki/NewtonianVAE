#!/usr/bin/env python3

from models import NewtonianVAE
from datasets.memory import ExperienceReplay

from dm_control import suite

import torch

import numpy as np

from matplotlib import pyplot as plt

from tensorboardX import SummaryWriter


batch_size = 20
max_episode = 100
max_sequence = 150
epoch_max = 100000

device = "cpu"
if torch.cuda.is_available():
  device = "cuda"

fig = plt.figure()


# env = suite.load(domain_name="point_mass", task_name="hard")
env = suite.load(domain_name="reacher", task_name="hard")


Train_Replay = ExperienceReplay(max_episode, max_sequence, 2, device)
Test_Replay = ExperienceReplay(1, max_sequence, 2, device)


print("Collect train data")
for episode in range(max_episode):
  print(f"Episode : {episode + 1}/{max_episode}", end="\r")
  time_step = env.reset()

  actions = []
  observations = []

  for _ in range(max_sequence):
    action = np.array([0.5+(np.random.rand()-0.5), np.random.rand()*0.25])
    time_step = env.step(action)
    video = env.physics.render(64, 64, camera_id=0)

    actions.append(action[np.newaxis, :])
    observations.append(video.transpose(2, 0, 1)[np.newaxis, :, :, :]/255.0)
  
    plt.clf()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(video)
    plt.pause(0.001)
  
  Train_Replay.append(np.concatenate(observations), np.concatenate(actions), episode)


print("Collect test data")
for episode in range(1):
  print(f"Episode : {episode + 1}/{max_episode}", end="\r")
  time_step = env.reset()

  actions = []
  observations = []

  for _ in range(max_sequence):
    action = np.array([0.5+(np.random.rand()-0.5), np.random.rand()*0.25])
    time_step = env.step(action)
    video = env.physics.render(64, 64, camera_id=0)

    actions.append(action[np.newaxis, :])
    observations.append(video.transpose(2, 0, 1)[np.newaxis, :, :, :]/255.0)
  
    plt.clf()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(video)
    plt.pause(0.001)
  
  Test_Replay.append(np.concatenate(observations), np.concatenate(actions), episode)


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
    time_step = env.reset()

    video = env.physics.render(64, 64, camera_id=0)
    I = torch.from_numpy(video.astype(np.float32).transpose(2, 0, 1)[np.newaxis, :, :, :]/255.0).to(device)
    v_t = torch.zeros((1, 2)).to(device)
    x_t = model.encoder.sample({"I": I})["x"]

    all_positions = []

    for i in range(max_sequence):
      u = np.array([0.5+(np.random.rand()-0.5), np.random.rand()*0.25]).astype(np.float32)
      time_step = env.step(u)
      u_t = torch.from_numpy(u[np.newaxis, :]).to(device)
    
      video = env.physics.render(64, 64, camera_id=0)
      I = torch.from_numpy(video.astype(np.float32).transpose(2, 0, 1)[np.newaxis, :, :, :]/255.0).to(device)

      I_t, x_t, v_t, I_next, x_next = model.infer(I, u_t, x_t, v_t)

      all_positions.append(x_t.to("cpu").detach().numpy()[0].tolist())

      plt.clf()
      ax1 = fig.add_subplot(2, 2, 1)
      ax2 = fig.add_subplot(2, 2, 2)
      ax3 = fig.add_subplot(2, 2, 3)
      ax4 = fig.add_subplot(2, 2, 4)
      ax1.imshow(video)
      ax2.imshow(np.clip(I_t.to("cpu").detach().numpy()[0].transpose(1, 2, 0), 0., 1.))
      ax3.imshow(np.clip(I_next.to("cpu").detach().numpy()[0].transpose(1, 2, 0), 0., 1.))
      ax4.set_xlim(-1.1, 1.1)
      ax4.set_ylim(-1.1, 1.1)
      ax4.scatter(np.array(all_positions)[:, 0], np.array(all_positions)[:, 1], s=0.1)
      plt.pause(0.001)
