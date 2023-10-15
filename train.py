#!/usr/bin/env python3
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

from models import ConditionalNewtonianVAE
from utils import visualize, memory, env


def data_loop(epoch, loader, model, device, train_mode=False):
    mean_loss = 0
    mean_correlation_x = 0
    mean_correlation_y = 0

    supervised_pos = []
    infered_pos = []

    for batch_idx, (I, D, u, p, label) in enumerate(tqdm(loader)):
        B, T, C = label.size()
        y = torch.ones(B, T, 1).to(device, non_blocking=True)
        R = p[:, :, :].reshape(B, T, 4)
        batch_size = I.size()[0]

        if train_mode:
            loss, pos = model.train({
                "I": I.to(device, non_blocking=True).permute(1, 0, 2, 3, 4),
                "u": u.to(device, non_blocking=True).permute(1, 0, 2), 
                "y": y.to(device, non_blocking=True).permute(1, 0, 2),
                "R": R.to(device, non_blocking=True).permute(1, 0, 2)})
        else:
            loss, pos = model.test({
                "I": I.to(device, non_blocking=True).permute(1, 0, 2, 3, 4),
                "u": u.to(device, non_blocking=True).permute(1, 0, 2), 
                "y": y.to(device, non_blocking=True).permute(1, 0, 2),
                "R": R.to(device, non_blocking=True).permute(1, 0, 2)})

        mean_loss += loss * batch_size

        supervised_pos.append(p[:, :-1].detach().cpu().numpy())
        infered_pos.append(pos.permute(1, 0, 2).detach().cpu().numpy())
        

        mean_correlation_x += np.corrcoef(
                np.concatenate(supervised_pos, axis=0).reshape(-1, 4)[:, 0],
                np.concatenate(infered_pos, axis=0).reshape(-1, 4)[:, 0])

        
        mean_correlation_y += np.corrcoef(
                np.concatenate(supervised_pos, axis=0).reshape(-1, 4)[:, 1],
                np.concatenate(infered_pos, axis=0).reshape(-1, 4)[:, 1])

    mean_correlation_x = mean_correlation_x[0, 1]
    mean_correlation_y = mean_correlation_y[0, 1]
    mean_loss /= len(loader.dataset)
    mean_correlation_x /= len(loader.dataset)
    mean_correlation_y /= len(loader.dataset)

    if train_mode:
        print('\nEpoch: {} Train loss: {:.4f} Correlation X: {} Y: {}'.format(epoch, mean_loss, mean_correlation_x, mean_correlation_y))
    else:
        print('\nEpoch: {}  Test loss: {:.4f} Correlation X: {} Y: {}'.format(epoch, mean_loss, mean_correlation_x, mean_correlation_y))
    return mean_loss, (mean_correlation_x, mean_correlation_y)


def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/train/point_mass.yml",
                        help='config path ex. config/sample/train/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_root_path = f"results/{timestamp}"
    save_weight_path = f"{save_root_path}/weights"
    save_video_path = f"{save_root_path}/videos"
    save_correlation_path = f"{save_root_path}/correlations"

    os.makedirs(save_root_path, exist_ok=True)
    os.makedirs(save_correlation_path, exist_ok=True)
    shutil.copy2(args.config, save_root_path+"/")

    #====================#
    # Define data loader #
    #====================#
    train_loader = memory.make_loader(cfg, "train")
    validation_loader = memory.make_loader(cfg, "validation")

    writer = SummaryWriter(comment="NewtonianVAE")

    #==============#
    # Define model #
    #==============#
    model = ConditionalNewtonianVAE(**cfg["model"])
    print(model)


    best_loss: float = 1e32

    with tqdm(range(1, cfg["epoch_size"]+1)) as pbar:

        for epoch in pbar:
            pbar.set_description(f"[Epoch {epoch}]")

            train_loss: float = 0.
            validation_loss: float = 0.
            test_loss: float = 0.

            #================#
            # Training phase #
            #================#
            train_loss, correlations = data_loop(epoch, train_loader,
                                   model, cfg["device"], train_mode=True)
            writer.add_scalar('train/loss', train_loss, epoch - 1)
            writer.add_scalar('train/correlation_x', correlations[0], epoch - 1)
            writer.add_scalar('train/correlation_y', correlations[1], epoch - 1)

            #==================#
            # Validation phase #
            #==================#
            validation_loss, correlations = data_loop(
                epoch, validation_loader, model, cfg["device"], train_mode=False)
            writer.add_scalar('validation/loss', validation_loss, epoch - 1)
            writer.add_scalar('validation/correlation_x', correlations[0], epoch - 1)
            writer.add_scalar('validation/correlation_y', correlations[1], epoch - 1)

            pbar.set_postfix({"validation": validation_loss,
                              "train": train_loss})


            #============#
            # Save model #
            #============#
            model.save(f"{save_weight_path}", f"{epoch}.weight")

            #=================#
            # Save best model #
            #=================#
            if validation_loss < best_loss:
                model.save(f"{save_weight_path}", f"best.weight")
                best_loss = validation_loss

            #==============#
            # Encode video #
            #==============#
            if epoch % cfg["check_epoch"] == 0:
                visualizer = visualize.Visualization()
                all_latent_position: list = []
                all_observation_position: list = []

                #============#
                # Test phase #
                #============#
                for idx, (I, D, u, p, label) in enumerate(validation_loader):
                    B, T, C = label.size()
                    y = torch.ones(B, T, 1).to(cfg["device"], non_blocking=True)

                    for step in range(0, cfg["dataset"]["train"]["sequence_size"]-1):

                        I_t, I_tp1, x_q_t, x_p_tp1 = model.estimate(
                            I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                            I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step],
                            D.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                            D.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step],
                            u.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[step+1],
                            y.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[step+1])


                        observation_position = p.permute(1, 0, 2)[step+1] - p.permute(1, 0, 2)[0]
                        all_observation_position.append(observation_position.to("cpu").detach().numpy()[0].tolist())
                        all_latent_position.append(x_q_t.to("cpu").detach().numpy()[0].tolist())

                        visualizer.append(
                            env.postprocess_observation(I[:, :, 0:3].permute(1, 0, 2, 3, 4)[step].to(
                                "cpu", non_blocking=True).detach().numpy()[0].transpose(1, 2, 0), cfg["bit_depth"]),
                            env.postprocess_observation(I_t.to("cpu", non_blocking=True).detach().to(
                                torch.float32).numpy()[0][0:3, :, :].transpose(1, 2, 0), cfg["bit_depth"]),
                            np.array(all_latent_position)
                        )

                visualizer.encode(save_video_path, f"{epoch}.{idx}.mp4")
                visualizer.add_images(writer, epoch)
                print()

if __name__ == "__main__":
    main()
