#!/usr/bin/env python3
import os
import pprint
import argparse
import yaml
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from models import ConditionalNewtonianVAE
from util import env
from environments import load, ControlSuiteEnv



def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/point_mass.yml",
                        help='config path e.g. config/sample/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as _file:
        cfg = yaml.safe_load(_file)
        pprint.pprint(cfg)

    
    save_root_path = cfg["weight"]["path"].replace("weights", "operability")
    os.makedirs(save_root_path, exist_ok=True)

    #================#
    # Define model   #
    #================#
    model = ConditionalNewtonianVAE(**cfg["model"])
    model.load(**cfg["weight"])

    success_errors = []
    failure_errors = []

    for episode in tqdm(range(0, 50)):
        fig, ((axis1, axis2), (axis3, axis4)) = plt.subplots(2, 2, tight_layout=True)

        frames = []
        #==================#
        # Get target image #
        #==================#
        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()

        number = time_step[2]
        label = torch.eye(10)[number].cuda().unsqueeze(0)

        target_observation, state, reward, done = _env.step(torch.zeros(1, 2))
        # target_x_q_t = model.encoder.sample_mean({"I_t": target_observation.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"]),
        #     "y_t": label})

        _env = ControlSuiteEnv(**cfg["environment"])
        time_step = _env.reset()
        
        number = time_step[2]
        label = torch.eye(10)[number].cuda().unsqueeze(0)

        action = torch.zeros(1, 2)

        axis1.cla()
        axis2.cla()
        axis3.cla()
        axis4.cla()

        for _ in range(400):

            #===================#
            # Get current image #
            #===================#
            observation, state, reward, done = _env.step(action.cpu())
            # _env.render()

            x_q_t = model.encoder.sample_mean({
                "I_t": observation.permute(2, 0, 1)[np.newaxis, :, :, :].to(cfg["device"])})
            reconstructed_image = model.decoder.sample_mean({"x_t": x_q_t})

            #============#
            # Get action #
            #============#
            # action = (target_x_q_t - x_q_t).detach()
            action = -x_q_t.detach()/25.

            # action = torch.flip(action, dims=[1])
            # action = -action
            # action[0, 1] = -action[0, 1]
            # action[0, 0] = -action[0, 0]

            # art1 = axis1.imshow(env.postprocess_observation(target_observation.detach().numpy(), 8))

            # art2 = axis2.imshow(env.postprocess_observation(observation.detach().numpy().copy(), 8))

            # axis3.set_ylim(-0.5, 0.5)
            # bar1, bar2 = axis3.bar(["X", "Y"], action[0].cpu().detach().numpy(), color=["black", "black"])

            # art4 = axis4.imshow(env.postprocess_observation(reconstructed_image[0].permute(1, 2, 0).cpu().detach().numpy(), 8))

            # frames.append([art1, art2, bar1, bar2, art4])


        error = np.sqrt(np.sum(state.observation["position"]**2))
        if error < 0.07:
            print(f"Success: {error}")
            success_errors.append(error)
        else:
            print(f"Failure: {error}")
            failure_errors.append(error)

        # ani = animation.ArtistAnimation(fig, frames, interval=10)
        # ani.save(f"{save_root_path}/output.episode_{episode}.error_{error}.mp4", writer="ffmpeg")
        plt.close()


    mean_error = np.mean(success_errors)
    std_error = np.std(success_errors)

    print(f"Success: {len(success_errors)}")
    print(f"Mean error: {mean_error}")
    print(f"Variance error: {std_error}")

    mean_error = np.mean(failure_errors)
    std_error = np.std(failure_errors)

    print(f"Failure: {len(failure_errors)}")
    print(f"Mean error: {mean_error}")
    print(f"Variance error: {std_error}")



if __name__ == "__main__":
    main()
