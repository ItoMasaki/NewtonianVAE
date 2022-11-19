#!/usr/bin/env python3
import numpy as np
import yaml
from matplotlib import pyplot as plt

from models.memory import ExperienceReplay
from environments import load


yaml_path = "config/sample/collect_point_mass_dataset.yml"

with open(yaml_path) as file:
    config = yaml.safe_load(file)

if config["env"] == "reacher_nvae":
  env = load(domain_name="reacher_nvae", task_name="hard", task_kwargs={"whole_range": False})
elif config["env"] == "reacher":
  env = load(domain_name="reacher", task_name="hard")
elif config["env"] == "point_mass":
  env = load(domain_name="point_mass", task_name="hard")
else:
  raise NotImplementedError(f"{config['env']}")


#############################
print("[*] Check an enviroment")
time_step = env.reset()

for _ in range(100):
  action = np.array([0.5+(np.random.rand()-0.5), 0.5+(np.random.rand()-0.5)/2])
  time_step = env.step(action)
  video = env.physics.render(64, 64, camera_id=0)

  plt.cla()
  plt.imshow(video)
  plt.pause(0.01)

#############################


for mode in ["train", "test"]:
  if config["env"] == "reacher_nvae":
    if mode == "test":
      env = load(domain_name="reacher_nvae", task_name="hard", task_kwargs={"whole_range": False})
    else:
      env = load(domain_name="reacher_nvae", task_name="hard", task_kwargs={"whole_range": True})

  elif config["env"] == "reacher":
    env = load(domain_name="reacher", task_name="hard")

  elif config["env"] == "point_mass":
    env = load(domain_name="point_mass", task_name="hard")

  else:
    raise NotImplementedError(f"{config['env']}")

  max_episode = config[mode]["max_episode"]
  max_sequence = config[mode]["max_sequence"]
  save_path = config[mode]["save_path"]
  save_filename = config[mode]["save_filename"]
  
  print(f"############## CONFIG PARAMS [{mode}] ##############")
  print(f"  max_episode : {max_episode}")
  print(f" max_sequence : {max_sequence}")
  print(f"    save_path : {save_path}")
  print(f"save_filename : {save_filename}")
  print(f"####################################################")
  
  
  save_memory = ExperienceReplay(max_episode, max_sequence, 2, "cpu")
  
  
  print("Collect data")
  for episode in range(max_episode):
    print(f"Episode : {episode + 1}/{max_episode}", end="\r")
    time_step = env.reset()
  
    actions = []
    observations = []

    direction = np.random.uniform(-1, 1, 2)

    for _ in range(max_sequence):
      if mode == "train":
        action = np.random.uniform(-1, 1, 2)
      else:
        action = np.array([np.random.normal(direction[0], 0.1), np.random.normal(direction[1]/2., 0.1)])
      time_step = env.step(action)
      video = env.physics.render(64, 64, camera_id=0)
  
      actions.append(action[np.newaxis, :])
      observations.append(video.transpose(2, 0, 1)[np.newaxis, :, :, :]/255.0)

      # plt.cla()
      # plt.imshow(video)
      # plt.pause(0.0001)
    
    save_memory.append(np.concatenate(observations), np.concatenate(actions), episode)
  
  
  print()
  save_memory.save(save_path, save_filename)
