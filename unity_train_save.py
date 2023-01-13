import numpy as np
from PIL import Image
import torch

#   initialize
im_top_array = torch.zeros([30,100,64,64,3], dtype=torch.int32)
im_side_array = torch.zeros([30,100,64,64,3], dtype=torch.int32)
im_hand_array = torch.zeros([30,100,64,64,3], dtype=torch.int32)
action = torch.zeros([30,100,3], dtype=torch.int32)

batch_size = 1
##########################################
### load images and convert to ndarray ###
##########################################

#   (size)im_top_array (1, 100, 64, 64, 3)
#   (episode, batch, width, height, rgb)
for num_episode in range(30):
    im_top_array[num_episode] = batch_size
    im_side_array[num_episode] = batch_size
    im_hand_array[num_episode] = batch_size
    for i in range(100):
        # top_image
        im_top_tmp = np.array(Image.open("datasets/train/top_image/image_000_" + str(i).zfill(4) + ".jpg"))

        # side image
        im_side_tmp = np.array(Image.open("datasets/train/side_image/image_000_" + str(i).zfill(4) + ".jpg"))

        # hand image
        im_hand_tmp = np.array(Image.open("datasets/train/hand_image/image_000_" + str(i).zfill(4) + ".jpg"))

        im_top_array[num_episode][i] = torch.tensor(im_top_tmp)
        im_side_array[num_episode][i] = torch.tensor(im_side_tmp)
        im_hand_array[num_episode][i] = torch.tensor(im_hand_tmp)


##################
### set action ###
##################
#   load action file
action_tmp = np.loadtxt("/home/admin_pc/NewtonianVAE/datasets/actions_train.txt")

#   (size)action (100, 100, 3)
for num_episode in range(30):
    action[num_episode] = batch_size
    for i in range(100):
        action[num_episode][i] = torch.tensor(action_tmp[i])

#   save npz file
np.savez("datasets/unity/train.npz", I_top=im_top_array, I_side=im_side_array, I_hand=im_hand_array, action=action)
# a
