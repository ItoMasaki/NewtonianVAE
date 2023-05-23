#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib import animation

import numpy as np


loaded_data = np.load('validation.npz')
print(loaded_data.files)

colors = loaded_data['colors']
print(colors.shape)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

frames = []

for batch in range(10):
    print(batch)
    for idx in range(100):
        frames.append([ax1.imshow(colors[batch, idx].transpose(1, 2, 0).astype(np.uint8))])


ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
ani.save('animation.mp4', writer='ffmpeg')
