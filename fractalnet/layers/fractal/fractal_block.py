from torch import nn
from conv_block import ConvBlock

class FractalBlock(nn.Module):
    def __init__(self, num_cols, c_in, c_out):
        super(FractalBlock, self).__init__()

        self.max_depth = 2**(num_cols-1)
        interval = self.max_depth
        cols = []
        self.cols_cnt_on_depth = [0] * self.max_depth
        for col in range(num_cols):
            cols.append([])
            for module_idx in range(self.max_depth):
                if (module_idx + 1) % interval == 0:
                    module = ConvBlock(
                        c_in=c_in if (module_idx + 1) == interval else c_out, #first in the col accepts the input, remaining ones the output dim
                        c_out=c_out
                    )
                else:
                    module = None
                cols[-1].append(module)

                self.cols_cnt_on_depth[module_idx] += 1
            interval /= 2

        self.cols = nn.ModuleList([nn.ModuleList(b) for b in cols])

    def forward(self, x):
        outs = [x] * len(self.cols)
        for depth in range(self.max_depth):
            outs_to_join = []
            left_col = len(self.cols) - self.cols_cnt_on_depth[depth]
            for col in range(left_col, len(self.cols)):
                cur_x = outs[col]
                cur_module = self.cols[col][depth]
                outs_to_join.append(cur_module(cur_x))

            joined = self.joiner(outs_to_join)
            for col in range(left_col, len(self.cols)):
                outs[col] = joined

        return outs[0]
