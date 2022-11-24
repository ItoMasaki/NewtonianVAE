#!/usr/bin/env python3

import argparse
import yaml
import numpy as np
import datetime
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models import NewtonianVAE
from utils import visualize, memory


parser = argparse.ArgumentParser(description='Collection dataset')
parser.add_argument('--config', type=str, help='config path ex. config/sample/train/point_mass.yml')
args = parser.parse_args()

with open(args.config) as file:
  cfg = yaml.safe_load(file)

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_root_path = f"results/{timestamp}"
save_weight_path = f"{save_root_path}/weights"
save_video_path = f"{save_root_path}/videos"


Train_Replay = memory.ExperienceReplay(cfg["max_episode"], cfg["max_sequence"], 2, cfg["device"])
Test_Replay = memory.ExperienceReplay(1, cfg["max_sequence"], 2, cfg["device"])
Train_Replay.load(cfg["train_path"])
Test_Replay.load(cfg["test_path"])

visualizer = visualize.Visualization()

writer = SummaryWriter(comment="NewtonianVAE")

model = NewtonianVAE(device=cfg["device"])

if cfg["init_params"]:
  model.init_params()

best_loss: float = 1e32
beta: float = 0.001


with tqdm(range(1, cfg["epoch_max"]+1)) as pbar:

  for epoch in pbar:
    pbar.set_description(f"[Epoch {epoch}]")

    for _ in range(200):
      I, u = Train_Replay.sample(cfg["batch_size"])
      train_loss = model.train({"I": I, "u": u, "beta": beta})
      writer.add_scalar('train_loss', train_loss, epoch)

      I, u = Test_Replay.sample(1)
      test_loss = model.test({"I": I, "u": u, "beta": beta})
      writer.add_scalar('test_loss', test_loss, epoch)

      pbar.set_postfix({"train": train_loss, "test": test_loss})

    if epoch%cfg["check_epoch"] == 0:
      model.save(f"{save_weight_path}", f"{epoch}.weight") 

      if train_loss < best_loss:
        model.save(f"{save_weight_path}", f"best.weight")
        best_loss = train_loss

      all_positions: list = []

      x_p_t = model.encoder.sample({"I_t": I[0]}, reparam=True)["x_t"]

      for step in range(0, cfg["max_sequence"]-1):

        I_t, I_tp1, x_q_t, x_p_tp1 = model.estimate(I[step+1], I[step], u[step+1])

        all_positions.append(x_q_t.to("cpu").detach().numpy()[0].tolist())

        visualizer.append(
                I[step].to("cpu").detach().numpy()[0].transpose(1, 2, 0),
                I_t.to("cpu").detach().to(torch.float32).numpy()[0].transpose(1, 2, 0),
                I_tp1.to("cpu").detach().to(torch.float32).numpy()[0].transpose(1, 2, 0),
                np.array(all_positions)
        )

        x_p_t = x_p_tp1

      visualizer.encode(save_video_path, f"{epoch}.mp4")

    if 30 <= epoch and epoch <= 60:
      beta += 0.0333
