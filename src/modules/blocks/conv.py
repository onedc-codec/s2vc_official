import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()
        self.act = nn.GELU()


    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        return h + self.skip(x)
