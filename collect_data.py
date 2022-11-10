#!/usr/bin/env python3
from dm_control import suite
import numpy as np
import yaml


from datasets.memory import ExperienceReplay


yaml_path = "config/make_dataset/sample.yml"

with open(yaml_path) as file:
    config = yaml.safe_load(file)


if config["env"] == "reacher2d":
  env = suite.load(domain_name="reacher", task_name="hard")
elif config["env"] == "point_mass":
  env = suite.load(domain_name="point_mass", task_name="hard")
else:
  raise NotImplementedError(f"{config['env']}")


max_episode = config["max_episode"]
max_sequence = config["max_sequence"]
save_path = config["save_path"]
save_filename = config["save_filename"]

print("############## CONFIG PARAMS ##############")
print(f"  max_episode : {max_episode}")
print(f" max_sequence : {max_sequence}")
print(f"    save_path : {save_path}")
print(f"save_filename : {save_filename}")
print("###########################################")


save_memory = ExperienceReplay(max_episode, max_sequence, 2, "cpu")


print("Collect data")
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
  
  save_memory.append(np.concatenate(observations), np.concatenate(actions), episode)


print()
save_memory.save(save_path, save_filename)
