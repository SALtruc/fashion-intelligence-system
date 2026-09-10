import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(network.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images: torch.Tensor):
        feature_map = self.features(images)
        embedding = self.pool(feature_map).flatten(1)
        return F.normalize(embedding, p=2, dim=1)


def build_cae_decoder():
    channels = [512, 256, 128, 64, 32]
    blocks = []

    for input_channels, output_channels in zip(channels[:-1], channels[1:]):
        blocks.extend([
            nn.ConvTranspose2d(
                input_channels, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        ])

    blocks.extend([
        nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid(),
    ])
    return nn.Sequential(*blocks)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, encoder: ResNet18Encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = build_cae_decoder()

    def forward(self, images: torch.Tensor):
        feature_map = self.encoder.features(images)
        embedding = self.encoder.pool(feature_map).flatten(1)
        reconstruction = self.decoder(feature_map)
        embedding = F.normalize(embedding, p=2, dim=1)
        return reconstruction, embedding
