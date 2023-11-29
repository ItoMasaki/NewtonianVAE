import os
import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import LinearSegmentedColormap

cdict = {
    "red": [
        (0.0, 0.0, 0.3),
        (0.25, 0.2, 0.4),
        (0.5, 0.8, 0.9),
        (0.75, 0.9, 1.0),
        (1.0, 0.4, 1.0),
    ],
    "green": [
        (0.0, 0.0, 0.2),
        (0.25, 0.2, 0.5),
        (0.5, 0.5, 0.8),
        (0.75, 0.8, 0.9),
        (1.0, 0.9, 1.0),
    ],
    "blue": [
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
}

cmap = LinearSegmentedColormap("custom", cdict, 12)


class Visualization:
    def __init__(self):
        self.frames: list = []
        self.reconstruction_images = []

    def append(self, I_t, rec_I_t, latent, points):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(14, 7))

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.ax5 = ax5
        self.ax6 = ax6

        latent_x_min = np.min(latent[:, :, 0])
        latent_x_max = np.max(latent[:, :, 0])

        latent_y_min = np.min(latent[:, :, 1])
        latent_y_max = np.max(latent[:, :, 1])

        latent_r_min = np.min(latent[:, :, 2])
        latent_r_max = np.max(latent[:, :, 2])

        points_x_min = np.min(points[:, :, 0])
        points_x_max = np.max(points[:, :, 0])

        points_y_min = np.min(points[:, :, 1])
        points_y_max = np.max(points[:, :, 1])

        points_r_min = np.min(points[:, :, 2])
        points_r_max = np.max(points[:, :, 2])

        ratio_latent = (latent_x_max - latent_x_min) / (latent_y_max - latent_y_min)
        ratio_x = (latent_x_max - latent_x_min) / (points_x_max - points_x_min)
        ratio_y = (latent_y_max - latent_y_min) / (points_y_max - points_y_min)
        ratio_r = (latent_r_max - latent_r_min) / (points_r_max - points_r_min)

        self.ax1.set_title(r"$ I_t $")
        self.ax1.axis('off')

        self.ax2.set_title(r"$ \hat{I}_t $")
        self.ax2.axis('off')

        self.ax3.set_title(r"$ trajectory $")
        self.ax3.set_aspect(ratio_latent, adjustable="box")


        self.ax4.set_title(r"$ latent - position : X $")
        self.ax4.set_aspect(ratio_x, adjustable="box")

        self.ax5.set_title(r"$ latent - position : Y $")
        self.ax5.set_aspect(ratio_y, adjustable="box")

        self.ax6.set_title(r"$ latent - position : R $")
        self.ax6.set_aspect(ratio_r, adjustable="box")

        episodes = I_t.shape[0]
        time_steps = I_t.shape[1]

        points_x = []
        points_y = []
        points_r = []

        latent_x = []
        latent_y = []
        latent_r = []

        colors = []

        for episode in range(episodes):
            for time_step in range(time_steps):
                art_1 = self.ax1.imshow(I_t[episode, time_step], cmap=cmap, vmin=0, vmax=1)
                art_2 = self.ax2.imshow(rec_I_t[episode, time_step], cmap=cmap, vmin=0, vmax=1)

                points_x.append(points[episode, time_step, 0])
                points_y.append(points[episode, time_step, 1])
                points_r.append(points[episode, time_step, 2])

                latent_x.append(latent[episode, time_step, 0])
                latent_y.append(latent[episode, time_step, 1])
                latent_r.append(latent[episode, time_step, 2])

                colors.append(time_step/time_steps)

                self.ax3.set_xlabel(r"$ X $")
                self.ax3.set_ylabel(r"$ Y $")
                art_3 = self.ax3.scatter(points_x, points_y, s=1, c=colors, vmin=0, vmax=1)

                self.ax4.set_xlabel(r"$ Latent - X $")
                self.ax4.set_ylabel(r"$ Position - X $")
                art_4 = self.ax4.scatter(latent_x, points_x, s=1, c=colors, vmin=0, vmax=1)

                self.ax5.set_xlabel(r"$ Latent - Y $")
                self.ax5.set_ylabel(r"$ Position - Y $")
                art_5 = self.ax5.scatter(latent_y, points_y, s=1, c=colors, vmin=0, vmax=1)

                self.ax6.set_xlabel(r"$ Latent - R $")
                self.ax6.set_ylabel(r"$ Position - R $")
                art_6 = self.ax6.scatter(latent_r, points_r, s=1, c=colors, vmin=0, vmax=1)

                self.frames.append([art_1, art_2, art_3, art_4, art_5, art_6])

    def encode(self, save_path, file_name):
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

        ani = ArtistAnimation(self.fig, self.frames, interval=100)
        ani.save(f"{save_path}/{file_name}", writer="ffmpeg")
        plt.cla()
        self.frames = []

        plt.close()

    def add_images(self, writer, epoch):
        writer.add_images("reconstruction_images", np.stack(self.reconstruction_images), epoch, dataformats="NHWC")
        self.reconstruction_images = []
