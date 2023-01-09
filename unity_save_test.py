import numpy as np
from PIL import Image
import torch

# initialize
im_top_array = torch.zeros([1,100,64,64,3], dtype=torch.int32)
im_side_array = torch.zeros([1,100,64,64,3], dtype=torch.int32)
im_hand_array = torch.zeros([1,100,64,64,3], dtype=torch.int32)
action = torch.zeros([1,100,3], dtype=torch.int32)

batch_size = 1
##########################################
### load images and convert to ndarray ###
##########################################

# (size)im_top_array (30, 1, 64, 64, 3)
# (episode, batch, width, height, rgb)
for num_episode in range(1):
    im_top_array[num_episode] = batch_size
    for i in range(100):
        # top_image
        im_top_tmp = np.array(Image.open("datasets/test/top_image/image_000_" + str(i).zfill(4) + ".jpg"))

        # side image
        im_side_tmp = np.array(Image.open("datasets/test/side_image/image_000_" + str(i).zfill(4) + ".jpg"))

        # hand image
        im_hand_tmp = np.array(Image.open("datasets/test/hand_image/image_000_" + str(i).zfill(4) + ".jpg"))

        im_top_array[num_episode][i] = torch.tensor(im_top_tmp)
        im_side_array[num_episode][i] = torch.tensor(im_side_tmp)
        im_hand_array[num_episode][i] = torch.tensor(im_hand_tmp)

# print(im_top_array.size())
# print(im_side_array.size())
# print(im_hand_array.size())


##################
### set action ###
##################
random_x = [0, 2, 3, 0, -5, -2, 4, 2, -2, -1, 5, 0, -4, 1, 4, 0, 3, 2, -3, -2, 3, 1, 1, 3, 1, 3, 2, 2, -1, 5, -3, 1, -1, 3, -1, 5, -2, -4, 1, -5, -1, 0, 2, -1, 1, 3, 5, -4, 5, 3, -2, -3, -4, -1, 1, -5, 4, -3, 1, 4, 1, -1, -4, -3, 1, -1, 0, 1, -3, -4, -4, -1, -2, 1, -5, -5, -3, 3, 3, -3, -3, -3, -4, -3, -3, -4, 0, 0, -1, -2, 4, 5, -1, 3, 0, -5, 2, -3, -1, -1]
random_y = [0, 3, -5, 3, -5, 0, -5, 0, -5, -4, -4, 4, -3, -5, -5, 3, -1, 3, 1, -2, -5, 0, -2, -1, -3, 0, 5, -5, 5, -2, 1, 1, -5, 0, -4, 1, -5, -5, 1, -4, 1, -3, 1, -5, 5, 2, -3, -3, 3, -5, 3, -1, -3, -1, 3, -1, -3, -5, -1, -2, 1, 2, -5, 5, 1, 1, -4, 1, 2, 0, 1, -4, -4, -2, 5, 4, -1, -5, -4, 0, 1, -5, -3, 1, 2, 1, 5, 5, 3, -4, -3, -4, 0, -3, 3, -4, 1, -3, 1, 1]
random_z = [0, 1, 5, 1, 6, 9, 4, 9, 3, 3, 2, 8, 5, 0, 0, 10, 9, 3, 6, 0, 4, 9, 10, 6, 6, 0, 8, 0, 3, 8, 2, 9, 4, 2, 1, 5, 6, 0, 0, 0, 10, 9, 4, 7, 3, 8, 6, 0, 10, 7, 2, 4, 10, 0, 6, 3, 7, 4, 1, 5, 10, 7, 3, 6, 2, 0, 10, 7, 8, 10, 5, 7, 9, 0, 2, 0, 1, 4, 8, 4, 6, 8, 6, 1, 10, 4, 7, 2, 8, 2, 9, 2, 7, 6, 1, 4, 8, 10, 4, 1]


# (size)action (30, 100)
for num_episode in range(1):
    action[num_episode] = batch_size
    for i in range(100):
        action_tmp = np.stack([random_x[i+(num_episode*100)], random_y[i+(num_episode*100)], random_z[i+(num_episode*100)]])
    action[num_episode] = torch.tensor(action_tmp)
# print(action.size())

# save npz file
np.savez("datasets/test.npz", I_top=im_top_array, I_side=im_side_array, I_hand=im_hand_array, action=action)