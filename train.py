#!/usr/bin/env python3

import yaml
import torch
import numpy as np
import datetime
from tqdm import tqdm
from dm_control import suite
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from tensorboardX import SummaryWriter

from components.models import NewtonianVAE
from components.memory import ExperienceReplay


cfg_path = "config/sample/train.yml"
with open(cfg_path) as file:
  cfg = yaml.safe_load(file)

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_weight_path = f"weights/{timestamp}"


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


Train_Replay = ExperienceReplay(cfg["max_episode"], cfg["max_sequence"], 2, cfg["device"])
Test_Replay = ExperienceReplay(1, cfg["max_sequence"], 2, cfg["device"])
Train_Replay.load(cfg["train_path"])
Test_Replay.load(cfg["test_path"])


writer = SummaryWriter(comment="NewtonianVAE")


model = NewtonianVAE(device=cfg["device"])
if cfg["init_params"]:
  model.init_params()

with tqdm(range(cfg["epoch_max"])) as pbar:

  for epoch in pbar:
    pbar.set_description(f"[Epoch {epoch}]")

    for _ in range(int(cfg["max_episode"]/cfg["batch_size"])):
      I, u = Train_Replay.sample(cfg["batch_size"])
      train_loss = model.train({"I": I, "u": u})
      writer.add_scalar('train_loss', train_loss, epoch)

      I, u = Test_Replay.sample(1)
      test_loss = model.test({"I": I, "u": u})
      writer.add_scalar('test_loss', test_loss, epoch)

      pbar.set_postfix({"train_loss": train_loss, "test_loss": test_loss})

    if epoch%cfg["check_epoch"] == 0: # and epoch != 0:
      model.save(f"{save_weight_path}", f"{epoch}.weight") 

      frames: list = []
      all_positions = []

      x_t = model.encoder.sample({"I": I[0]})["x"]

      for step in range(0, cfg["max_sequence"]-1):

        I_t, x_t, v_t, I_next, x_next = model.infer(I[step+1], u[step], x_t)

        all_positions.append(x_t.to("cpu").detach().numpy()[0].tolist())

        plt.cla()
        # plt.clf()

        artists_1 = ax1.imshow(I[step].to("cpu").detach().numpy()[0].transpose(1, 2, 0))
        artists_2 = ax2.imshow(np.clip(I_t.to("cpu").detach().numpy()[0].transpose(1, 2, 0), 0., 1.))
        artists_3 = ax3.imshow(np.clip(I_next.to("cpu").detach().numpy()[0].transpose(1, 2, 0), 0., 1.))
        # artists_4 = ax4.set_xlim(-10.1, 10.1)
        # artists_4 = ax4.set_ylim(-10.1, 10.1)
        artists_4 = ax4.scatter(np.array(all_positions)[:, 0], np.array(all_positions)[:, 1], s=0.9)

        plt.pause(0.01)

        frames.append([artists_1, artists_2, artists_3, artists_4])

      ani = ArtistAnimation(fig, frames, interval=100)
      ani.save('animation.mp4', writer='ffmpeg')
