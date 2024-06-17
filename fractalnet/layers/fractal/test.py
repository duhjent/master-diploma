from torch import nn

from conv_block import ConvBlock

class FractalExpansion(nn.Module):
    def __init__(self, f_ctor):
        super(FractalExpansion, self).__init__()
        self.left = ConvBlock()
        self.right = nn.Sequential(
            f_ctor(),
            f_ctor()
        )
