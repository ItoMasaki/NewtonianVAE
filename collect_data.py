import numpy as np
import yaml

from dm_control import suite

from datasets.memory import ExperienceReplay


batch_size = 20
max_episode = 100
max_sequence = 150
epoch_max = 100000


# env = suite.load(domain_name="point_mass", task_name="hard")
env = suite.load(domain_name="reacher", task_name="hard")


Train_Replay = ExperienceReplay(max_episode, max_sequence, 2, "cpu")
Test_Replay = ExperienceReplay(1, max_sequence, 2, "cpu")


print("Collect train data")
for episode in range(max_episode):
  print(f"Episode : {episode + 1}/{max_episode}", end="\r")
  time_step = env.reset()

  actions = []
  observations = []

  for _ in range(max_sequence):
    action = np.array([0.5+(np.random.rand()-0.5), np.random.rand()*0.25])
    time_step = env.step(action)
    video = env.physics.render(64, 64, camera_id=0)

    actions.append(action[np.newaxis, :])
    observations.append(video.transpose(2, 0, 1)[np.newaxis, :, :, :]/255.0)
  
  Train_Replay.append(np.concatenate(observations), np.concatenate(actions), episode)


print("Collect test data")
for episode in range(1):
  print(f"Episode : {episode + 1}/{max_episode}", end="\r")
  time_step = env.reset()

  actions = []
  observations = []

  for _ in range(max_sequence):
    action = np.array([0.5+(np.random.rand()-0.5), np.random.rand()*0.25])
    time_step = env.step(action)
    video = env.physics.render(64, 64, camera_id=0)

    actions.append(action[np.newaxis, :])
    observations.append(video.transpose(2, 0, 1)[np.newaxis, :, :, :]/255.0)
  
  Test_Replay.append(np.concatenate(observations), np.concatenate(actions), episode)
