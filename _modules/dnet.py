from torch.nn import Module, Conv1d, ReLU, BatchNorm1d, AvgPool1d, Sigmoid
from torch.nn import Sequential, Linear, Dropout, AdaptiveAvgPool1d
from torch import Tensor, mean, cat
import torch

from typing import List

class MeanChannel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = lambda x: mean(x, 2)
        
    def forward(self,
                x: Tensor) \
                    -> Tensor:
        return self.f(x)

class ExpandConv(Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = lambda x: mean(x, 2)
        
    def forward(self,
                x: Tensor) \
                    -> Tensor:
        return self.f(x)

class FilterBankConv1d(Module):
    def __init__(self,
                 stride_res: int = 320,
                 n_filters: int = 2) -> None:
        super().__init__()
        self.conv_32 = Conv1d(
            1,
            n_filters,
            kernel_size=32,
            stride=stride_res,
            bias=False,
        )
        
        self.conv_64 = Conv1d(
            1,
            n_filters,
            kernel_size=64,
            stride=stride_res,
            bias=False,
        )

        self.conv_128 = Conv1d(
            1,
            n_filters,
            kernel_size=128,
            stride=stride_res,
            bias=False,
        )

        self.conv_256 = Conv1d(
            1,
            n_filters,
            kernel_size=256,
            stride=stride_res,
            bias=False,
        )
    
    def forward(self,
                x: Tensor) \
                    -> Tensor:
      out1 = self.conv_32(x)
      out_len = out1.shape[2] - 4

      x = cat((x[..., 64:], x[..., :64]), dim=2)
      out2 = self.conv_64(x)
      
      x = cat((x[..., 128:], x[..., :128]), dim=2)
      out3 = self.conv_128(x)
      
      x = cat((x[..., 256:], x[..., :256]), dim=2)
      out4 = self.conv_256(x)

      return cat((out1[..., :out_len], out2[..., :out_len], out3[..., :out_len], out4[..., :out_len]), dim=1)

class SEBlock(Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.squeeze = AdaptiveAvgPool1d(1)
        self.excitation = Sequential(
            Linear(c, c // r, bias=False),
            ReLU(),
            Linear(c // r, c, bias=False),
            Sigmoid()
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
 
        return x * y.expand_as(x)

class DNetConvBlock(Module):
    def __init__(self,
                in_channels: int,
                depth_multiplier: float = 1.0,
                cat: bool = False,
                squeeze: bool = True) -> None:
        super().__init__()
        self.cat = cat
        layers: List = []

        non_linearity = ReLU()

        conv: Conv1d = Conv1d(
            in_channels,
            int(in_channels*depth_multiplier),
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False
        )

        layers.append(conv)
        layers.append(BatchNorm1d(int(in_channels*depth_multiplier)))
        layers.append(non_linearity)

        conv: Conv1d = Conv1d(
            int(in_channels*depth_multiplier),
            int(in_channels*depth_multiplier),
            kernel_size=3,
            stride=1,
            padding="same",
            groups=int(in_channels*depth_multiplier),
            bias=False
        )

        layers.append(conv)
        layers.append(BatchNorm1d(int(in_channels*depth_multiplier)))
        layers.append(non_linearity)

        conv: Conv1d = Conv1d(
            int(in_channels*depth_multiplier),
            in_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False
        )

        layers.append(conv)
        layers.append(BatchNorm1d(in_channels))
        layers.append(non_linearity)

        if squeeze:
            layers.append(SEBlock(in_channels))

        layers.append(Dropout(p=0))

        self.block: Sequential = Sequential(
            *layers
        )

    def forward(self,
                x: Tensor) \
                    -> Tensor:
        inp = x
        x = self.block(x)
        
        if self.cat:
            out = cat((x, inp), dim=1)
        else:
            out = x + inp
        
        return out

class DNet(Module):
    @staticmethod
    def get_block(in_channels: int,
                  depth_multiplier: int,
                  cat: bool = False,
                  downsample: bool = False,
                  squeeze: bool = False) -> List:
        block = []
                
        block.append(
            DNetConvBlock(in_channels=in_channels,
                          depth_multiplier=depth_multiplier,
                          cat=cat)
        )
        
        if downsample:
            block.append(AvgPool1d(2))

        return block

    def __init__(self,
                 stride_res: int = 320,
                 depth_multiplier: float = 1.0,
                 num_blocks: int = 4,
                 downsampling_layers: List = [4],
                 squeeze: bool = True,
                 n_filters: int = 2) -> None:
        super().__init__()

        layers: List = []
        non_linearity = ReLU()

        first_conv = FilterBankConv1d(
            stride_res=stride_res,
            n_filters=n_filters
        )

        # first_conv = FilterBankSinc(
        #     stride_res=stride_res
        # )
        
        layers.append(first_conv)
        # layers.append(BatchNorm1d(16))
        # layers.append(non_linearity)

        layers += self.get_block(n_filters*4,
                                 depth_multiplier,
                                 cat=False,
                                 downsample=True,
                                 squeeze=squeeze)
        layers += self.get_block(n_filters*4,
                                 depth_multiplier,
                                 cat=True,
                                 downsample=False,
                                 squeeze=squeeze)
        
        downsampled = 0
        for block_idx in range(4, num_blocks+1):
            if block_idx in downsampling_layers:
                layers += self.get_block(int(n_filters*8*2**downsampled),
                                         depth_multiplier,
                                         cat=True,
                                         downsample=True,
                                         squeeze=squeeze)
                downsampled += 1
            else:
                layers += self.get_block(int(n_filters*8*2**downsampled),
                                         depth_multiplier,
                                         cat=False,
                                         downsample=False,
                                         squeeze=squeeze)
        layers.append(MeanChannel())
        layers.append(Linear(int(n_filters*8*2**downsampled), 10))
        
        self.cnn: Sequential = Sequential(
            *layers
        )

    def forward(self, x):
        return self.cnn(x)
    
class DNetADV(Module):
    @staticmethod
    def get_block(in_channels: int,
                  depth_multiplier: int,
                  cat: bool = False,
                  downsample: bool = False,
                  squeeze: bool = False) -> List:
        block = []
                
        block.append(
            DNetConvBlock(in_channels=in_channels,
                          depth_multiplier=depth_multiplier,
                          cat=cat)
        )
        
        if downsample:
            block.append(AvgPool1d(2))

        return block

    def __init__(self,
                 stride_res: int = 320,
                 depth_multiplier: float = 1.0,
                 num_blocks: int = 4,
                 downsampling_layers: List = [4],
                 squeeze: bool = True,
                 n_filters: int = 2) -> None:
        super().__init__()

        layers: List = []
        non_linearity = ReLU()

        first_conv = FilterBankConv1d(
            stride_res=stride_res,
            n_filters=n_filters
        )
        
        layers.append(first_conv)
        # layers.append(BatchNorm1d(16))
        # layers.append(non_linearity)

        # layers.append(PhiNet()) FINIRE QUI
        
        self.cnn: Sequential = Sequential(
            *layers
        )

    def forward(self, x):
        return self.cnn(x)
    
if __name__ == "__main__":
    a = DNetADV()
    x = torch.randn(1, 1, 64000)
    
    print(a(x).shape)