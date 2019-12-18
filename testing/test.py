from model.gan import Generator
from model.gan import Discriminator
import torch


dummy_x = torch.ones(size=(1, 100, 1, 1))
generator = Generator(latent_size=100)
discriminator = Discriminator(feature_map_size=64)

generator = generator.to("cpu")
discriminator = discriminator.to("cpu")

gen_y = generator(dummy_x)
gen_y.shape


dis_y = discriminator(gen_y)
print(dis_y)





