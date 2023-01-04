from models.model import NewtonianVAE
import torch

encoder={"input_dim": 3, "output_dim": 3, "act_func_name": "ReLU"}
decoder={"input_dim": 3, "output_dim": 3, "act_func_name": "ReLU", "device" : "cuda"}
transition={"delta_time": 0.1}
velocity={"batch_size": 10, "delta_time": 0.1, "act_func_name": "ReLU", "device": "cuda", "use_data_efficiency": 0}
NewtonianVAE(encoder, decoder, transition, velocity)