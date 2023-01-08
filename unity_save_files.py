import numpy as np
from PIL import Image

im_top_array = []
im_side_array = []
im_hand_array = []

# load images and convert to ndarray
for i in range(100):
    im_top = np.array(Image.open("datasets/top_image/image_000_" + str(i).zfill(4) + ".jpg"))
    im_top_array.append(im_top)

    im_side = np.array(Image.open("datasets/side_image/image_000_" + str(i).zfill(4) + ".jpg"))
    im_side_array.append(im_side)

    im_hand = np.array(Image.open("datasets/hand_image/image_000_" + str(i).zfill(4) + ".jpg"))
    im_hand_array.append(im_hand)

# set action
random_x = [0, 1, -3, 2, 5, 5, 2, -2, 5, 5, 5, 4, -1, 2, -1, 4, 1, -2, 1, -4, 3, 1, -1, 2, -3, 5, 2, 0, -1, 2, 3, 2, 4, 5, 4, -1, -2, -1, -1, 3, -3, 0, 1, 2, -4, 1, -1, -1, 3, -5, 2, 5, 2, -2, 4, -3, 3, -3, 5, 2, 1, 1, 3, -3, -2, 2, 0, 1, 0, 4, -5, -1, 2, 5, -4, 4, -1, 1, -2, -4, -4, 0, 4, -2, -5, 1, -5, 5, 2, 0, 3, -5, 1, 3, -2, -3, -2, -5, -3, -2]
random_y = [0, -2, 0, 0, 4, 5, 3, 0, 1, -5, 2, 0, 5, 2, 0, 4, -1, 2, 0, 2, 2, -1, -1, 4, 2, -1, 1, -1, 3, -5, 0, 3, 4, 0, -4, 1, -5, -1, -2, -4, 2, 4, 5, -2, 4, 0, -1, -3, 1, 4, -5, 0, -2, -4, -4, 4, -3, 0, 1, 5, -2, -1, 1, 0, 3, -1, -4, 0, 1, 4, 3, 1, 1, -2, -5, -3, -5, -1, -2, -3, -1, -4, -5, -2, 2, -1, 2, -1, 2, -4, -5, -2, 1, 5, -5, 2, 2, 3, -1, -2]
random_z = [0, 2, 9, 0, 10, 5, 9, 5, 6, 4, 9, 8, 4, 0, 4, 8, 3, 6, 4, 0, 9, 0, 6, 10, 10, 6, 1, 8, 1, 0, 0, 6, 9, 1, 4, 2, 1, 7, 7, 2, 5, 2, 6, 5, 9, 7, 8, 7, 2, 5, 0, 6, 2, 6, 6, 2, 0, 6, 10, 4, 5, 8, 5, 7, 3, 9, 1, 4, 4, 10, 2, 2, 8, 7, 0, 0, 10, 10, 5, 8, 10, 2, 8, 9, 0, 0, 3, 2, 1, 3, 9, 2, 3, 9, 8, 10, 3, 9, 1, 0]
action = np.stack([random_x, random_y, random_z])

# save npz file
np.savez("datasets/train.npz", I_top=im_top_array, I_side=im_side_array, I_hand=im_hand_array, action=action)
