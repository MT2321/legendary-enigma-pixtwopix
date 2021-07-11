# U-NET discriminator
import torch
import torch.nn as nn
from config import CONV2D_CONFIG_1, CONV2D_CONFIG_2, LEAKYRELU_SLOPE

class CNNUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cnn_config: dict) -> None:
        super().__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **cnn_config),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(LEAKYRELU_SLOPE)
        )

    def forward(self, x):
        return self.conv_unit(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        first_layer = nn.Sequential(
            nn.Conv2d(2*in_channels, 64, **CONV2D_CONFIG_1), nn.LeakyReLU(0.2))
        down1 = CNNUnit(64, 128, CONV2D_CONFIG_1)
        down2 = CNNUnit(128, 256, CONV2D_CONFIG_1)
        down3 = CNNUnit(256, 512, CONV2D_CONFIG_2)
        patches_layer = CNNUnit(512, 1, CONV2D_CONFIG_2)
        self.model = nn.Sequential(
            first_layer, down1, down2, down3, patches_layer)

    def forward(self, x, y):
        # Concatenate along the channels axis
        concat_images = torch.cat([x, y], dim=1)
        return self.model(concat_images)


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    print(model)
    preds = model.forward(x, y)
    print(preds.shape)


if __name__ == '__main__':
    test()
