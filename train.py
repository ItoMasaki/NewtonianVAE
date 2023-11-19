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

    for batch_idx, (I, u, p, label) in enumerate(tqdm(loader)):
        label = torch.eye(2)[label.int()].to(device, non_blocking=True).squeeze(2)
        R = p[:, :, 2].unsqueeze(2)
        batch_size = I.size()[0]

        E, T, C, H, W = I.size() # 1, 10, 3, 64, 64
        # 1, 1, 2 -> 1, 10, 2
        label = label.repeat(1, T, 1)

        if train_mode:
            kl_loss, decoder_loss, output = model(
                I.to(device, non_blocking=True).permute(1, 0, 2, 3, 4),
                u.to(device, non_blocking=True).permute(1, 0, 2), 
                label.to(device, non_blocking=True).permute(1, 0, 2),
                R.to(device, non_blocking=True).permute(1, 0, 2)
                ) * batch_size


            kl_loss += kl_loss
            decoder_loss += decoder_loss
            mean_loss += kl_loss + decoder_loss

            corr = 1 - correlation(
                    output[0].detach().cpu().numpy().flatten(),
                    p[0].detach().cpu().numpy().flatten()
                    )
            total_corr += corr
        else:
            kl_loss, decoder_loss, output = model(
                I.to(device, non_blocking=True).permute(1, 0, 2, 3, 4),
                u.to(device, non_blocking=True).permute(1, 0, 2), 
                label.to(device, non_blocking=True).permute(1, 0, 2),
                R.to(device, non_blocking=True).permute(1, 0, 2)
                ) * batch_size

            kl_loss += kl_loss
            decoder_loss += decoder_loss
            mean_loss += kl_loss + decoder_loss

            corr = 1 - correlation(
                    output[0].detach().cpu().numpy().flatten(),
                    p[0].detach().cpu().numpy().flatten()
                    )
            total_corr += corr

    mean_loss /= len(loader.dataset)
    total_corr /= len(loader.dataset)

    print("kl_loss: ", kl_loss, "decoder_loss: ", decoder_loss, "total_corr: ", total_corr)

    if train_mode:
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, mean_loss))
    else:
        print('Test loss: {:.4f}'.format(mean_loss))
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
    # beta: float = 1.

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
            writer.add_scalar('train_loss', train_loss, epoch - 1)

            #==================#
            # Validation phase #
            #==================#
            validation_loss = data_loop(
                epoch, validation_loader, model, cfg["device"], beta, train_mode=False)
            writer.add_scalar('validation_loss', validation_loss, epoch - 1)

            pbar.set_postfix({"validation": validation_loss,
                              "train": train_loss})


            #============#
            # Save model #
            #============#
            torch.save(model.state_dict(), f"{save_weight_path}/{epoch}.weight")

            #=================#
            # Save best model #
            #=================#
            if validation_loss < best_loss:
                torch.save(model.state_dict(), f"{save_weight_path}/best.weight")
                best_loss = validation_loss

            if 30 <= epoch and epoch < 60:
                beta += 0.0333

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
                for idx, (I, u, p, label) in enumerate(test_loader):

                    label = torch.eye(2)[label.int()].to(cfg["device"], non_blocking=True).squeeze(2)

                    for step in range(0, cfg["dataset"]["train"]["sequence_size"]-1):
                        pass

                        # I_t, I_tp1, x_q_t, x_p_tp1 = model.estimate(
                        #     I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                        #     I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step],
                        #     u.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[step+1],
                        #     label.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[0])
                        # I_t, I_tp1, x_q_t, x_p_tp1 = model.estimate(
                        #     I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                        #     I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step],
                        #     u.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[step+1],
                        #     y)


                        # latent_position = model.encoder.sample_mean({
                        #     "I_t": I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                        #     "y_t": label.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[0]})
                        # latent_position = model.encoder.sample_mean({
                        #     "I_t": I.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                        #     "y_t": y})

                        # all_latent_position.append(
                        #     latent_position.to("cpu").detach().numpy()[0].tolist())

                        # observation_position = p.permute(1, 0, 2)[step+1] - p.permute(1, 0, 2)[0]
                        # all_observation_position.append(observation_position.to("cpu").detach().numpy()[0].tolist())

                        # visualizer.append(
                        #     env.postprocess_observation(I.permute(1, 0, 2, 3, 4)[step].to(
                        #         "cpu", non_blocking=True).detach().numpy()[0].transpose(1, 2, 0), cfg["bit_depth"]),
                        #     env.postprocess_observation(I_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[
                        #         0].transpose(1, 2, 0), cfg["bit_depth"]),
                        #     np.array(all_latent_position)
                        # )

                    # np.savez(f"{save_correlation_path}/{epoch}.{idx}", {'latent': all_latent_position, 'observation': all_observation_position})


                # visualizer.encode(save_video_path, f"{epoch}.{idx}.mp4")
                # visualizer.add_images(writer, epoch)
                # print()

                # correlation_X = np.corrcoef(
                #     np.array(all_observation_position)[:, 0], np.array(all_latent_position)[:, 0])
                # correlation_Y = np.corrcoef(
                #     np.array(all_observation_position)[:, 1], np.array(all_latent_position)[:, 1])
                # correlation_R = np.corrcoef(
                #     np.array(all_observation_position)[:, 2], np.array(all_latent_position)[:, 2])
                # print("X", correlation_X[0, 1], "Y", correlation_Y[0, 1], "R", correlation_R[0, 1])

                # writer.add_scalar('correlation/X', correlation_X[0, 1], epoch - 1)
                # writer.add_scalar('correlation/Y', correlation_Y[0, 1], epoch - 1)
                # writer.add_scalar('correlation/R', correlation_R[0, 1], epoch - 1)


if __name__ == "__main__":
    main()
