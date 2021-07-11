# U-NET
import torch
import torch.nn as nn
from config import CONV2D_CONFIG_1_G, LEAKYRELU_SLOPE, CONV_TRANSPOSE_2D_CONFIG_1
# Encoder building blocks
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalize=True, leaky=True):
        super().__init__()
        self.layers = [] 
        self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **CONV2D_CONFIG_1_G))
        if batch_normalize:
            self.layers.append(nn.BatchNorm2d(out_channels))
        if leaky:
            self.layers.append(nn.LeakyReLU(LEAKYRELU_SLOPE))
        else:
            self.layers.append(nn.ReLU())

        self.conv_down = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.conv_down(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=True, batch_normalize=True, activation='relu'):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **CONV_TRANSPOSE_2D_CONFIG_1)))
        if batch_normalize:
            self.layers.append(nn.BatchNorm2d(out_channels))
        elif activation == 'relu':
            self.layers.append(nn.ReLU())
        elif activation == 'Tanh':
            self.layers.append(nn.Tanh())

        self.conv_up = nn.Sequential(*self.layers)
        if use_dropout:
            self.use_dropout = use_dropout
            self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        return self.conv_up(x)


class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self):
        super().__init__()
    
        #Encoder
        self.initial_down = DownBlock(in_channels=3, out_channels=64, batch_normalize=False)
        self.down1 = DownBlock(in_channels=64, out_channels=128)
        self.down2 = DownBlock(in_channels=128, out_channels=256)
        self.down3 = DownBlock(in_channels=256, out_channels=512)
        self.down4 = DownBlock(in_channels=512, out_channels=512)
        self.down5 = DownBlock(in_channels=512, out_channels=512)
        self.down6 = DownBlock(in_channels=512, out_channels=512)
        # self.down7 = DownBlock(in_channels=512, out_channels=512)
        #Bottleneck
        self.bottleneck = DownBlock(in_channels=512, out_channels=512, batch_normalize=False)
        #Decoder
        self.up1 = UpBlock(in_channels=512, out_channels=512, use_dropout=True)
        self.up2 = UpBlock(in_channels=1024, out_channels=512, use_dropout=True)
        self.up3 = UpBlock(in_channels=1024, out_channels=512, use_dropout=True)
        self.up4 = UpBlock(in_channels=1024, out_channels=512)
        self.up5 = UpBlock(in_channels=1024, out_channels=256)
        self.up6 = UpBlock(in_channels=512, out_channels=128)
        self.up7 = UpBlock(in_channels=256, out_channels=64)

        # self.output_layer = UpBlock(in_channels=128, out_channels=3, activation='Tanh', batch_normalize=False)
        self.output_layer  = nn.Sequential(nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
                    nn.Tanh())
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([d7, u1], 1))
        u3 = self.up3(torch.cat([d6, u2], 1))
        u4 = self.up4(torch.cat([d5, u3], 1))
        u5 = self.up5(torch.cat([d4, u4], 1))
        u6 = self.up6(torch.cat([d3, u5], 1))
        u7 = self.up7(torch.cat([d2, u6], 1))
        return self.output_layer(torch.cat([u7, d1], 1))


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    print(model)
    preds = model.forward(x)
    print(preds.shape)


if __name__ == '__main__':
    test()


