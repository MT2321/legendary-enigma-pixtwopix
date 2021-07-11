from torch.nn.modules import padding
from torch.nn.modules.activation import LeakyReLU


CONV2D_CONFIG_1 = {'kernel_size': 4,
                   'stride': 2,
                   'padding': 1,
                   'padding_mode': 'reflect',
                   'bias': False}

CONV2D_CONFIG_2 = {'kernel_size': 4,
                   'stride': 1,
                   'padding': 1,
                   'padding_mode': 'reflect',
                   'bias': False}


CONV2D_CONFIG_1_G = {'kernel_size': 4,
                     'stride': 2,
                     'padding_mode': 'reflect',
                     'padding': 1,
                     'bias': False}

CONV_TRANSPOSE_2D_CONFIG_1 = {'kernel_size': 4,
                              'stride': 2,
                              'bias': False,
                              'padding':1
                              }


LEAKYRELU_SLOPE = 0.2
