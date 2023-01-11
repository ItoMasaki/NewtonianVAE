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

from models import NewtonianVAE
from utils import visualize, memory, env


def data_loop(epoch, loader, model, device, train_mode=False):
    mean_loss = 0

    for batch_idx, (I_top, I_side, I_hand, u) in enumerate(tqdm(loader)):
        batch_size = 1

        # if train_mode:
        #     mean_loss += model.train({"I_top_t": I_top.to(device, non_blocking=True).permute(1, 0, 4, 2, 3), "I_side_t": I_side.to(device, non_blocking=True).permute(1, 0, 4, 2, 3), "I_hand_t": I_hand.to(device, non_blocking=True).permute(1, 0, 4, 2, 3), "u": u.to(
        #         device, non_blocking=True).permute(1, 0, 2)}) * batch_size
        # else:
        #     mean_loss += model.test({"I_top_t": I_top.to(device, non_blocking=True).permute(1, 0, 4, 2, 3), "I_side_t": I_side.to(device, non_blocking=True).permute(1, 0, 4, 2, 3), "I_hand_t": I_hand.to(device, non_blocking=True).permute(1, 0, 4, 2, 3), "u": u.to(
        #         device, non_blocking=True).permute(1, 0, 2)}) * batch_size
        if train_mode:
            mean_loss += model.train({"I_top_t": I_top.to(device, non_blocking=True).permute(1, 0, 2, 3, 4), "I_side_t": I_side.to(device, non_blocking=True).permute(1, 0, 2, 3, 4), "I_hand_t": I_hand.to(device, non_blocking=True).permute(1, 0, 2, 3, 4), "u": u.to(
                device, non_blocking=True).permute(1, 0, 2)}) * batch_size
        else:
            mean_loss += model.test({"I_top_t": I_top.to(device, non_blocking=True).permute(1, 0, 2, 3, 4), "I_side_t": I_side.to(device, non_blocking=True).permute(1, 0, 2, 3, 4), "I_hand_t": I_hand.to(device, non_blocking=True).permute(1, 0, 2, 3, 4), "u": u.to(
                device, non_blocking=True).permute(1, 0, 2)}) * batch_size
        # if train_mode:
        #     mean_loss += model.train({"I_top_t": I_top.to(device, non_blocking=True), "I_side_t": I_side.to(device, non_blocking=True), "I_hand_t": I_hand.to(device, non_blocking=True), "u": u.to(device, non_blocking=True), "beta": beta}) * batch_size
        # else:
        #     mean_loss += model.test({"I_top_t": I_top.to(device, non_blocking=True), "I_side_t": I_side.to(device, non_blocking=True), "I_hand_t": I_hand.to(device, non_blocking=True), "u": u.to(device, non_blocking=True), "beta": beta}) * batch_size


    mean_loss /= len(loader.dataset)

    if train_mode:
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, mean_loss))
    else:
        print('Test loss: {:.4f}'.format(mean_loss))
    return mean_loss


def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/mopoe/mopoe_train.yml",
                        help='config path ex. config/mopoe/mopoe_train.yml')
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

    #====================#
    # Define data loader #
    #====================#
    train_loader = memory.make_loader(cfg, "train")
    validation_loader = memory.make_loader(cfg, "validation")
    test_loader = memory.make_loader(cfg, "test")

    visualizer_top = visualize.Visualization()
    visualizer_side = visualize.Visualization()
    visualizer_hand = visualize.Visualization()
    writer = SummaryWriter(comment="NewtonianVAE")

    #==============#
    # Define model #
    #==============#
    model = NewtonianVAE(**cfg["model"])


    best_loss: float = 1e32
    # beta: float = 0.001

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
                                   model, cfg["device"], train_mode=True)
            writer.add_scalar('train_loss', train_loss, epoch - 1)

            #==================#
            # Validation phase #
            #==================#
            validation_loss = data_loop(
                epoch, validation_loader, model, cfg["device"], train_mode=False)
            writer.add_scalar('validation_loss', validation_loss, epoch - 1)

            pbar.set_postfix({"validation": validation_loss,
                              "train": train_loss})

            #============#
            # Test phase #
            #============#
            for idx, (I_top, I_side, I_hand, u) in enumerate(test_loader):
                continue

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

            # if 30 <= epoch and epoch < 60:
            #     beta += 0.0333

            #==============#
            # Encode video #
            #==============#
            if epoch % cfg["check_epoch"] == 0:
            # if epoch == 1:

                all_positions: list = []

                for step in range(0, cfg["dataset"]["train"]["sequence_size"]-1):

                    I_top_t, I_side_t, I_hand_t, I_top_tp1, I_side_tp1, I_hand_tp1, x_q_t, x_p_tp1 = model.estimate(
                        # I_top.to(cfg["device"], non_blocking=True).permute(1, 0, 4, 2, 3)[step+1],
                        # I_side.to(cfg["device"], non_blocking=True).permute(1, 0, 4, 2, 3)[step+1],
                        # I_hand.to(cfg["device"], non_blocking=True).permute(1, 0, 4, 2, 3)[step+1],
                        # I_top.to(cfg["device"], non_blocking=True).permute(1, 0, 4, 2, 3)[step],
                        # I_side.to(cfg["device"], non_blocking=True).permute(1, 0, 4, 2, 3)[step],
                        # I_hand.to(cfg["device"], non_blocking=True).permute(1, 0, 4, 2, 3)[step],
                        # u.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[step+1])
                        I_top.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                        I_side.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                        I_hand.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step+1],
                        I_top.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step],
                        I_side.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step],
                        I_hand.to(cfg["device"], non_blocking=True).permute(1, 0, 2, 3, 4)[step],
                        u.to(cfg["device"], non_blocking=True).permute(1, 0, 2)[step+1])

                    all_positions.append(
                        x_q_t.to("cpu").detach().numpy()[0].tolist())

                    visualizer_top.append(
                        # env.postprocess_observation(I_top.permute(1, 0, 4, 2, 3)[step].to(
                        #     "cpu", non_blocking=True).detach().numpy()[0].transpose(1, 2, 0), cfg["bit_depth"]),
                        # env.postprocess_observation(I_top_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[
                        #     0].transpose(1, 2, 0), cfg["bit_depth"]),
                        # np.array(all_positions)
                        env.postprocess_observation(I_top.permute(1, 0, 2, 3, 4)[step].to(
                            "cpu", non_blocking=True).detach().numpy()[0].transpose(1, 2, 0), cfg["bit_depth"]),
                        env.postprocess_observation(I_top_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[
                            0].transpose(1, 2, 0), cfg["bit_depth"]),
                        np.array(all_positions)
                    )

                    visualizer_side.append(
                        # env.postprocess_observation(I_side.permute(1, 0, 4, 2, 3)[step].to(
                        #     "cpu", non_blocking=True).detach().numpy()[0].transpose(1, 2, 0), cfg["bit_depth"]),
                        # env.postprocess_observation(I_side_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[
                        #     0].transpose(1, 2, 0), cfg["bit_depth"]),
                        # np.array(all_positions)
                        env.postprocess_observation(I_side.permute(1, 0, 2, 3, 4)[step].to(
                            "cpu", non_blocking=True).detach().numpy()[0].transpose(1, 2, 0), cfg["bit_depth"]),
                        env.postprocess_observation(I_side_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[
                            0].transpose(1, 2, 0), cfg["bit_depth"]),
                        np.array(all_positions)
                    )

                    visualizer_hand.append(
                        # env.postprocess_observation(I_hand.permute(1, 0, 4, 2, 3)[step].to(
                        #     "cpu", non_blocking=True).detach().numpy()[0].transpose(1, 2, 0), cfg["bit_depth"]),
                        # env.postprocess_observation(I_hand_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[
                        #     0].transpose(1, 2, 0), cfg["bit_depth"]),
                        # np.array(all_positions)
                        env.postprocess_observation(I_hand.permute(1, 0, 2, 3, 4)[step].to(
                            "cpu", non_blocking=True).detach().numpy()[0].transpose(1, 2, 0), cfg["bit_depth"]),
                        env.postprocess_observation(I_hand_t.to("cpu", non_blocking=True).detach().to(torch.float32).numpy()[
                            0].transpose(1, 2, 0), cfg["bit_depth"]),
                        np.array(all_positions)
                    )

                visualizer_top.encode(save_video_path, "top" + f"{epoch}.{idx}.mp4")
                visualizer_top.add_images(writer, epoch)
                visualizer_side.encode(save_video_path, "side" + f"{epoch}.{idx}.mp4")
                visualizer_side.add_images(writer, epoch)
                visualizer_hand.encode(save_video_path, "hand" + f"{epoch}.{idx}.mp4")
                visualizer_hand.add_images(writer, epoch)
                print()


if __name__ == "__main__":
    main()
