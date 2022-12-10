#!/usr/bin/env python3.10
import pprint
import os
import argparse
import datetime
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import yaml

from models import NewtonianVAE
from utils import visualize, memory


parser = argparse.ArgumentParser(description='Collection dataset')
parser.add_argument('--config', type=str, default="config/sample/train/point_mass.yml",
                    help='config path ex. config/sample/train/point_mass.yml')
args = parser.parse_args()

with open(args.config) as file:
    cfg = yaml.safe_load(file)
    pprint.pprint(cfg)

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_root_path = f"results/{timestamp}"
save_weight_path = f"{save_root_path}/weights"
save_video_path = f"{save_root_path}/videos"

os.makedirs(save_root_path, exist_ok=True)
shutil.copy2(args.config, save_root_path+"/")

train_replay = memory.ExperienceReplay(**cfg["dataset"]["train"]["memory"])
validation_replay = memory.ExperienceReplay(
    **cfg["dataset"]["validation"]["memory"])
test_replay = memory.ExperienceReplay(**cfg["dataset"]["test"]["memory"])

train_replay.load(**cfg["dataset"]["train"]["data"])
validation_replay.load(**cfg["dataset"]["validation"]["data"])
test_replay.load(**cfg["dataset"]["test"]["data"])

train_loader = DataLoader(
    train_replay, **cfg["dataset"]["train"]["loader"])
validation_loader = DataLoader(
    test_replay, **cfg["dataset"]["validation"]["loader"])
test_loader = DataLoader(test_replay, **cfg["dataset"]["test"]["loader"])

visualizer = visualize.Visualization()

writer = SummaryWriter(comment="NewtonianVAE")

model = NewtonianVAE(**cfg["model"])
# model.init_params()

if cfg["load_model"]:
    model.load(cfg["load_model_path"], cfg["load_model_file"])


test_loss: float = 0.
best_loss: float = 1e32
beta: float = 0.001

with tqdm(range(1, cfg["epoch_size"]+1)) as pbar:

    for epoch in pbar:
        pbar.set_description(f"[Epoch {epoch}]")

        # for idx, (I, u, _) in enumerate(train_loader):
        for idx, (I, u) in enumerate(train_loader):
            train_loss = model.train(
                {"I": I.permute(1, 0, 2, 3, 4), "u": u.permute(1, 0, 2), "beta": beta})
            writer.add_scalar('train_loss', train_loss, epoch - 1)
            pbar.set_postfix({"test": test_loss, "train": train_loss})

        # for I, u, _ in test_loader:
        for I, u in validation_loader:
            test_loss = model.test(
                {"I": I.permute(1, 0, 2, 3, 4), "u": u.permute(1, 0, 2), "beta": beta})
            writer.add_scalar('test_loss', test_loss, epoch - 1)
            pbar.set_postfix({"test": test_loss, "train": train_loss})

        model.save(f"{save_weight_path}", f"{epoch}.weight")
        model.save_ckpt(f"{save_weight_path}", f"train.ckpt", epoch, test_loss)

        if test_loss < best_loss:
            model.save(f"{save_weight_path}", f"best.weight")
            best_loss = test_loss

        if 30 <= epoch and epoch < 60:
            beta += 0.0333

        if epoch % cfg["check_epoch"] == 0:

            for I, u in test_loader:

                all_positions: list = []

                for step in range(0, cfg["dataset"]["train"]["sequence_size"]-1):

                    I_t, I_tp1, x_q_t, x_p_tp1 = model.estimate(I.permute(1, 0, 2, 3, 4)[
                                                                step+1], I.permute(1, 0, 2, 3, 4)[step], u.permute(1, 0, 2)[step+1])

                    all_positions.append(
                        x_q_t.to("cpu").detach().numpy()[0].tolist())

                    visualizer.append(
                        I.permute(1, 0, 2, 3, 4)[step].to(
                            "cpu").detach().numpy()[0].transpose(1, 2, 0),
                        I_t.to("cpu").detach().to(torch.float32).numpy()[
                            0].transpose(1, 2, 0),
                        np.array(all_positions)
                    )

                visualizer.encode(save_video_path, f"{epoch}.mp4")
                print()
