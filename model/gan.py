
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class Generator(BaseModel):
    def __init__(self, latent_size=100, feature_map_size=64, num_channels=3):
        super().__init__()
        self.latent_size = latent_size
        self.feature_map_size = feature_map_size
        self.num_channels = num_channels
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_size,
                               self.feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 8),
            nn.ReLU(True),
            # state size. (self.feature_map_size*8) x 4 x 4
            nn.ConvTranspose2d(self.feature_map_size * 8,
                               self.feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 4),
            nn.ReLU(True),
            # state size. (self.feature_map_size*4) x 8 x 8
            nn.ConvTranspose2d(self.feature_map_size * 4,
                               self.feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 2),
            nn.ReLU(True),
            # state size. (self.feature_map_size*2) x 16 x 16
            nn.ConvTranspose2d(self.feature_map_size * 2,
                               self.feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map_size),
            nn.ReLU(True),
            # state size. (self.feature_map_size) x 32 x 32
            nn.ConvTranspose2d(self.feature_map_size,
                               self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.generator(input)
        return output


class Discriminator(BaseModel):
    def __init__(self, feature_map_size=64, num_channels=3):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.num_channels = num_channels

        self.discriminator = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.feature_map_size,
                      4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.feature_map_size) x 32 x 32
            nn.Conv2d(self.feature_map_size,
                      self.feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.feature_map_size*2) x 16 x 16
            nn.Conv2d(self.feature_map_size * 2,
                      self.feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.feature_map_size*4) x 8 x 8
            nn.Conv2d(self.feature_map_size * 4,
                      self.feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.feature_map_size*8) x 4 x 4
            nn.Conv2d(self.feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.discriminator(input)
        return output.view(-1, 1).squeeze(1)

def GAN(BaseModel):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
