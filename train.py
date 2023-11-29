#!/usr/bin/env python3
import pprint
import os
import argparse
import datetime
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm
import shutil
import yaml

from models import ConditionalNewtonianVAE
from utils import visualize, memory, env

# Correlation
from scipy.spatial.distance import correlation



def data_loop(epoch, loader, model, device, beta, train_mode=False):
    mean_loss = 0
    kl_loss = 0
    decoder_loss = 0
    total_corr = 0.

    with tqdm(loader, desc=f"[Epoch {epoch}]") as pbar:

        for batch_idx, (I, u, p, label) in enumerate(pbar):
            batch_size = I.size()[0]
            episode, time, channel, height, width = I.size() # 1, 10, 3, 64, 64

            # Rotation
            R = p[:, :, 2].unsqueeze(2)

            # Condition
            label = torch.eye(2)[label.int()].to(device, non_blocking=True).squeeze(2)
            # 1, 1, 2 -> 1, 10, 2
            C = label.repeat(1, time, 1)

            if train_mode:
                x_p_t = torch.zeros(batch_size, channel).to(device)
                loss, output_dict = model.train({
                    'v_t': u.to(device, non_blocking=True).permute(1, 0, 2),
                    'y_t': C.to(device, non_blocking=True).permute(1, 0, 2),
                    'x_tn1_p': x_p_t,
                    'I_t': I.to(device, non_blocking=True).permute(1, 0, 2, 3, 4),
                    'R_t': R.to(device, non_blocking=True).permute(1, 0, 2),
                }, return_dict=True)

                mean_loss += loss.item()
                kl_loss += output_dict["AnalyticalKullbackLeibler"].item()

                pbar.set_postfix({
                    "Loss": f"{ int(mean_loss/(batch_idx+1)) }",
                    "KL": f"{ kl_loss/(batch_idx+1) }",
                })
            else:
                x_p_t = torch.zeros(batch_size, channel).to(device)
                loss, output_dict = model.test({
                    'v_t': u.to(device, non_blocking=True).permute(1, 0, 2),
                    'y_t': C.to(device, non_blocking=True).permute(1, 0, 2),
                    'x_tn1_p': x_p_t,
                    'I_t': I.to(device, non_blocking=True).permute(1, 0, 2, 3, 4),
                    'R_t': R.to(device, non_blocking=True).permute(1, 0, 2),
                }, return_dict=True)

                mean_loss += loss.item()
                kl_loss += output_dict["AnalyticalKullbackLeibler"].mean().item()

                pbar.set_postfix({
                    "Loss": f"{ int(mean_loss/(batch_idx+1)) }",
                    "KL": f"{ kl_loss/(batch_idx+1) }",
                })

        mean_loss /= len(loader.dataset)
        total_corr /= len(loader.dataset)

        return mean_loss


def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/train/point_mass.yml",
                        help='config path ex. config/sample/train/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)
        pprint.pprint(cfg)

    # timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    timestamp = "test"
    save_root_path = f"results/{timestamp}"
    save_weight_path = f"{save_root_path}/weights"
    save_video_path = f"{save_root_path}/videos"
    save_correlation_path = f"{save_root_path}/correlations"

    os.makedirs(save_root_path, exist_ok=True)
    os.makedirs(save_correlation_path, exist_ok=True)
    os.makedirs(save_weight_path, exist_ok=True)
    shutil.copy2(args.config, save_root_path+"/")

    #====================#
    # Define data loader #
    #====================#
    train_loader = memory.make_loader(cfg, "train")
    validation_loader = memory.make_loader(cfg, "validation")
    test_loader = memory.make_loader(cfg, "test")

    writer = SummaryWriter(comment="NewtonianVAE")

    #==============#
    # Define model #
    #==============#
    model = ConditionalNewtonianVAE(**cfg["model"])
    print(model)


    best_loss: float = 1e32
    beta: float = 0.001
    corr_x: float = 0.
    corr_y: float = 0.
    corr_r: float = 0.

    with tqdm(range(1, cfg["epoch_size"]+1)) as pbar:

        for epoch in pbar:
            pbar.set_description(f"[Epoch {epoch}]")

            train_loss: float = 0.
            validation_loss: float = 0.
            test_loss: float = 0.

            #================#
            # Training phase #
            #================#
            train_loss = data_loop(epoch, train_loader,
                                   model, cfg["device"], beta, train_mode=True)

            #==================#
            # Validation phase #
            #==================#
            validation_loss = data_loop(
                epoch, validation_loader, model, cfg["device"], beta, train_mode=False)

            #============#
            # Save model #
            #============#
            model.save(f"{save_weight_path}/{epoch}.weight")

            #=================#
            # Save best model #
            #=================#
            if validation_loss < best_loss:
                model.save(f"{save_weight_path}/best.weight")
                best_loss = validation_loss

            #==============#
            # Encode video #
            #==============#
            if epoch % cfg["check_epoch"] == 0:
                visualizer = visualize.Visualization()

                all_latent_position: list = []
                all_position: list = []
                all_raw_images: list = []
                all_rec_images: list = []

                #============#
                # Test phase #
                #============#
                for idx, (I, u, p, label) in enumerate(test_loader):

                    one_episode_latent_position: list = []
                    one_episode_position: list = []
                    one_episode_raw_images: list = []
                    one_episode_rec_images: list = []

                    label = torch.eye(2)[label.int()].to(cfg["device"], non_blocking=True).squeeze(2)

                    p_t = np.zeros_like(p[0, 0, :].cpu().detach().numpy())

                    for step in range(0, cfg["dataset"]["train"]["sequence_size"]-1):
                        with torch.no_grad():
                            x_q_t = model.encoder.sample_mean({
                                "I_t": I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step],
                                "y_t": label.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[0],
                            })
                            I_t = model.decoder.sample_mean({
                                "x_t_p": x_q_t,
                                "y_t": label.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[0],
                            })

                        x_q_t = x_q_t.cpu().detach().numpy().copy()
                        # p_t = p.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[step]
                        # p_t = p_t.cpu().detach().numpy().copy()
                        p_t += 0.5 * u.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[step].cpu().detach().numpy().squeeze()
                        # print(p_t, x_q_t)

                        one_episode_latent_position.append(x_q_t)
                        one_episode_position.append(p_t.copy())
                        one_episode_raw_images.append(env.postprocess_observation(I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step].cpu().detach().numpy(), cfg["bit_depth"]))
                        one_episode_rec_images.append(env.postprocess_observation(I_t.cpu().detach().numpy(), cfg["bit_depth"]))

                    all_latent_position.append(one_episode_latent_position)
                    all_position.append(one_episode_position)
                    all_raw_images.append(one_episode_raw_images)
                    all_rec_images.append(one_episode_rec_images)

                all_latent_position = np.array(all_latent_position).squeeze()
                all_position = np.array(all_position).squeeze()
                all_raw_images = np.array(all_raw_images).squeeze().transpose(0, 1, 3, 4, 2)
                all_rec_images = np.array(all_rec_images).squeeze().transpose(0, 1, 3, 4, 2)

                _all_latent_position = all_latent_position.copy()
                _all_position = all_position.copy()

                mean_corr_x = 0.
                mean_corr_y = 0.
                mean_corr_r = 0.

                for idx in range(0, _all_latent_position.shape[0]):
                    mean_corr_x += np.corrcoef(_all_latent_position[idx, 0], _all_position[idx, 0])[0, 1]
                    mean_corr_y += np.corrcoef(_all_latent_position[idx, 1], _all_position[idx, 1])[0, 1]
                    mean_corr_r += np.corrcoef(_all_latent_position[idx, 2], _all_position[idx, 2])[0, 1]

                mean_corr_x /= _all_latent_position.shape[0]
                mean_corr_y /= _all_latent_position.shape[0]
                mean_corr_r /= _all_latent_position.shape[0]

                visualizer.append(
                        all_raw_images,
                        all_rec_images,
                        all_latent_position,
                        all_position,
                )
                visualizer.encode(save_video_path, f"{epoch}.{idx}.mp4")

            print()
            pbar.set_postfix({
                "correlation x": f"{mean_corr_x:.3f}",
                "correlation y": f"{mean_corr_y:.3f}",
                "correlation r": f"{mean_corr_r:.3f}",
            })


if __name__ == "__main__":
    main()
