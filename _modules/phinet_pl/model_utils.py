import torch.nn as nn
import torch


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling

    Args:
        input_shape ([tuple/int]): [Input size]
        kernel_size ([tuple/int]): [Kernel size]

    Returns:
        [tuple]: [Padding coeffs]
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    # input(kernel_size[0])
    if kernel_size[0] == 3:
        return (1, 1, 1, 1)
    elif kernel_size[0] == 5:
        return (3, 3, 3, 3)
    else:
        return (0, 0, 0, 0)

def preprocess_input(x, **kwargs):
    """Normalise channels between [-1, 1]

    Args:
        x ([Tensor]): [Contains the image, number of channels is arbitrary]

    Returns:
        [Tensor]: [Channel-wise normalised tensor]
    """

    return (x / 128.) - 1

def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    """Compute expansion factor based on the formula from the paper

    Args:
        t_zero ([int]): [initial expansion factor]
        beta ([int]): [shape factor]
        block_id ([int]): [id of the block]
        num_blocks ([int]): [number of blocks in the network]

    Returns:
        [float]: [computed expansion factor]
    """
    return (t_zero * beta) * block_id / num_blocks + t_zero * (num_blocks - block_id) / num_blocks

class ReLUMax(torch.nn.Module):
    def __init__(self, max):
        super(ReLUMax, self).__init__()
        self.max = max
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        return torch.clamp(self.relu(x), max = self.max)

class HSwish(torch.nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()
    
    def forward(self, x):
        return x * nn.ReLU6(inplace=True)(x + 3) / 6

class SEBlock(torch.nn.Module):
    """Implements squeeze-and-excitation block"""

    def __init__(self, in_channels, out_channels, h_swish=True):
        """Constructor of SEBlock

        Args:
            in_channels ([int]): [Input number of channels]
            out_channels ([int]): [Output number of channels]
            h_swish (bool, optional): [Whether to use the h_swish or not]. Defaults to True.
        """
        super(SEBlock, self).__init__()

        self.glob_pooling = lambda x: nn.functional.avg_pool2d(x, x.size()[2:])

        self.se_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
        )

        self.se_conv2 = nn.Conv2d(
            out_channels,
            in_channels,
            kernel_size=1,
            bias=False,
            padding="same"
        )

        if h_swish:
            self.activation = HSwish()
        else:
            self.activation = ReLUMax(6)

    def forward(self, x):
        """Executes SE Block

        Args:
            x ([Tensor]): [input tensor]

        Returns:
            [Tensor]: [output of squeeze-and-excitation block]
        """
        inp = x
        x = self.glob_pooling(x)
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)
        x = x.expand_as(inp) * inp

        return x


class DepthwiseConv2d(torch.nn.Conv2d):
    """Depthwise 2D conv

    Args:
        torch ([Tensor]): [Input tensor for convolution]
    """

    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )


class SeparableConv2d(torch.nn.Module):
    """Implements SeparableConv2d"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=torch.nn.functional.relu,
                 kernel_size=3,
                 stride=1,
                 padding=0, 
                 dilation=1,
                 bias=False,
                 padding_mode='zeros',
                 depth_multiplier=1,
                 ):
        """Constructor of SeparableConv2d

        Args:
            in_channels ([int]): [Input number of channels]
            out_channels ([int]): [Output number of channels]
            kernel_size (int, optional): [Kernel size]. Defaults to 3.
            stride (int, optional): [Stride for conv]. Defaults to 1.
            padding (int, optional): [Padding for conv]. Defaults to 0.
            dilation (int, optional): []. Defaults to 1.
            bias (bool, optional): []. Defaults to False.
            padding_mode (str, optional): []. Defaults to 'zeros'.
            depth_multiplier (int, optional): [Depth multiplier]. Defaults to 1.
        """
        super().__init__()

        self._layers = torch.nn.ModuleList()

        depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding="valid",
            dilation=1,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        spatialConv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            dilation=dilation,
            # groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )

        bn = torch.nn.BatchNorm2d(
            out_channels,
            eps=1e-3,
            momentum=0.999,
            affine=False,
        )
        
        self._layers.append(depthwise)
        self._layers.append(spatialConv)
        self._layers.append(bn)
        self._layers.append(activation)

    def forward(self, x):
        """Executes SeparableConv2d block

        Args:
            x ([Tensor]): [Input tensor]

        Returns:
            [Tensor]: [Output of convolution]
        """
        for l in self._layers:
            x = l(x)

        return x
