#!/usr/bin/env python3

import yaml
import numpy as np
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models import NewtonianVAE
from utils import visualize, memory


cfg_path = "config/sample/train.yml"
with open(cfg_path) as file:
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


with tqdm(range(1, cfg["epoch_max"]+1)) as pbar:

  for epoch in pbar:
    pbar.set_description(f"[Epoch {epoch}]")

    for _ in range(int(cfg["max_episode"]/cfg["batch_size"])):
      I, u = Train_Replay.sample(cfg["batch_size"])
      train_loss = model.train({"I": I, "u": u})
      writer.add_scalar('train_loss', train_loss, epoch)

      I, u = Test_Replay.sample(1)
      test_loss = model.test({"I": I, "u": u})
      writer.add_scalar('test_loss', test_loss, epoch)

      pbar.set_postfix({"train": train_loss, "test": test_loss})

    if epoch%cfg["check_epoch"] == 0:
      model.save(f"{save_weight_path}", f"{epoch}.weight") 

      if train_loss < best_loss:
        model.save(f"{save_weight_path}", f"best.weight")
        best_loss = train_loss

      all_positions: list = []

      for step in range(0, cfg["max_sequence"]-1):

        I_t, I_tp1, x_q_t, x_p_tp1 = model.estimate(I[step+1], I[step], u[step+1])

        all_positions.append(x_q_t.to("cpu").detach().numpy()[0].tolist())

        visualizer.append(
                I[step].to("cpu").detach().numpy()[0].transpose(1, 2, 0),
                I_t.to("cpu").detach().numpy()[0].transpose(1, 2, 0),
                I_tp1.to("cpu").detach().numpy()[0].transpose(1, 2, 0),
                np.array(all_positions)
        )

      visualizer.encode(save_video_path, f"{epoch}.mp4")
