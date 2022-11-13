#!/usr/bin/env python3

import datetime
import torch
import numpy as np
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from dm_control import suite

from components.models import NewtonianVAE
from components.memory import ExperienceReplay


batch_size = 50
max_episode = 1000
max_sequence = 100
epoch_max = 300
train_path = "datasets/sample/train.npz"
test_path = "datasets/sample/test.npz"
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_weight_path = f"weights/{timestamp}"

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
model.init_params()

for epoch in range(epoch_max):

  for _ in range(int(max_episode/batch_size)):
    I, u = Train_Replay.sample(batch_size)
    train_loss = model.train({"I": I, "u": u})
    print(f"Epoch : {epoch} train_loss : {train_loss}")
    writer.add_scalar('train_loss', train_loss, epoch)

    I, u = Test_Replay.sample(1)
    test_loss = model.test({"I": I, "u": u})
    print(f"Epoch : {epoch}  test_loss : {test_loss}", end="\033[1A\033[1000D")
    writer.add_scalar('test_loss', test_loss, epoch)

  if epoch%10 == 0 and epoch != 0:
    model.save(f"{save_weight_path}", f"{epoch}.weight") 

    print("\n")

    v_t = torch.zeros((1, 2)).to(device)
    x_t = model.encoder.sample({"I": I[0]})["x"]

    all_positions = []

    for step in range(0, max_sequence-1):

      I_t, x_t, v_t, I_next, x_next = model.infer(I[step+1], u[step], x_t, v_t)

      all_positions.append(x_t.to("cpu").detach().numpy()[0].tolist())

      plt.clf()

      ax1 = fig.add_subplot(2, 2, 1)
      ax2 = fig.add_subplot(2, 2, 2)
      ax3 = fig.add_subplot(2, 2, 3)
      ax4 = fig.add_subplot(2, 2, 4)

      ax1.imshow(I[step].to("cpu").detach().numpy()[0].transpose(1, 2, 0))
      ax2.imshow(np.clip(I_t.to("cpu").detach().numpy()[0].transpose(1, 2, 0), 0., 1.))
      ax3.imshow(np.clip(I_next.to("cpu").detach().numpy()[0].transpose(1, 2, 0), 0., 1.))
      # ax4.set_xlim(-10.1, 10.1)
      # ax4.set_ylim(-10.1, 10.1)
      ax4.scatter(np.array(all_positions)[:, 0], np.array(all_positions)[:, 1], s=0.9)

      plt.pause(0.001)
