from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in,
                              out_channels=c_out,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        self.bn = nn.BatchNorm2d(c_out)

        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        self.actv(out)

        return out
