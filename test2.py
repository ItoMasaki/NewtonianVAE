from pixyz import distributions as dist
from models.distributions import Encoder1, Encoder2


encoder1 = Encoder1(3,3,"ReLU")
encoder2 = Encoder2(3,3,"ReLU")

x_topside_t = dist.ProductOfNormal([encoder1, encoder2])

print("x_topside_t=", x_topside_t)