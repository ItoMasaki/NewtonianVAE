import numpy as np
from PIL import Image
import torch

#   initialize
im_top_array = torch.zeros([15,100,64,64,3], dtype=torch.int32)
im_side_array = torch.zeros([15,100,64,64,3], dtype=torch.int32)
im_hand_array = torch.zeros([15,100,64,64,3], dtype=torch.int32)
action = torch.zeros([15,100,3], dtype=torch.int32)

batch_size = 1
##########################################
### load images and convert to ndarray ###
##########################################

#   (size)im_top_array (1, 100, 64, 64, 3)
#   (episode, batch, width, height, rgb)
for num_episode in range(15):
    im_top_array[num_episode] = batch_size
    im_side_array[num_episode] = batch_size
    im_hand_array[num_episode] = batch_size
    for i in range(100):
        #   top_image
        im_top_tmp = np.array(Image.open("datasets/validation/top_image/image_000_" + str(i).zfill(4) + ".jpg"))

        #   side image
        im_side_tmp = np.array(Image.open("datasets/validation/side_image/image_000_" + str(i).zfill(4) + ".jpg"))

        #   hand image
        im_hand_tmp = np.array(Image.open("datasets/validation/hand_image/image_000_" + str(i).zfill(4) + ".jpg"))

        im_top_array[num_episode][i] = torch.tensor(im_top_tmp)
        im_side_array[num_episode][i] = torch.tensor(im_side_tmp)
        im_hand_array[num_episode][i] = torch.tensor(im_hand_tmp)


##################
### set action ###
##################
#   load action file
action_tmp = np.loadtxt("/home/admin_pc/NewtonianVAE/datasets/actions_validation.txt")

#   (size)action (1, 100, 3)
for num_episode in range(15):
    action[num_episode] = batch_size
    for i in range(100):
        action[num_episode][i] = torch.tensor(action_tmp[i])

#   save npz file
np.savez("datasets/unity/validation.npz", I_top=im_top_array, I_side=im_side_array, I_hand=im_hand_array, action=action)
# a