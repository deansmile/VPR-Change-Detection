import torch
import torch.nn as nn


# fmt: off
class MTF(nn.Module):
    """
    Original merge temporal feature
    Copied from C3PO, commit: 7aadc692c3313f7212b1c0f8114f7a2005419617
    https://github.com/DoctorKey/C-3PO/blob/main/src/model/Backbone.py
    """
    def __init__(self, channel, mode='iade', kernel_size=1, **kwargs):
        super(MTF, self).__init__()
        assert mode in ['i', 'a', 'd', 'e', 'ia', 'id', 'ie', 'iae', 'ide', 'iad', 'iade', 'i2ade', 'iad2e', 'i2ad2e', 'i2d']
        self.mode = mode
        self.channel = channel
        self.relu = nn.ReLU(inplace=True)
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        if 'i2' in mode:
            self.i0 = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            self.i1 = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.conv = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)

        if 'ad2'in mode:
            self.app = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            self.dis = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.res = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)

        self.exchange = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        print("MTF: mode: {} kernel_size: {}".format(self.mode, kernel_size))

    def forward(self, f0, f1):
        #t0 = self.conv(f0)
        #t1 = self.conv(f1)
        if 'i2' in self.mode:
            info = self.i0(f0) + self.i1(f1)
        else:
            info = self.conv(f0 + f1)

        if 'd' in self.mode:
            if 'ad2' in self.mode:
                disappear = self.dis(self.relu(f0 - f1))
            else:
                disappear = self.res(self.relu(f0 - f1))
        else:
            disappear = 0

        if 'a' in self.mode:
            if 'ad2' in self.mode:
                appear = self.app(self.relu(f1 - f0))
            else:
                appear = self.res(self.relu(f1 - f0))
        else:
            appear = 0

        if 'e' in self.mode:
            exchange = self.exchange(torch.max(f0, f1) - torch.min(f0, f1))
        else:
            exchange = 0

        if self.mode == 'i':
            f = info
        elif self.mode == 'a':
            f = appear
        elif self.mode == 'd':
            f = disappear
        elif self.mode == 'e':
            f = exchange
        elif self.mode == 'ia':
            f = info + 2 * appear
        elif self.mode in ['id', 'i2d']:
            f = info + 2 * disappear
        elif self.mode == 'ie':
            f = info + 2 * exchange
        elif self.mode == 'iae':
            f = info + appear + exchange
        elif self.mode == 'ide':
            f = info + disappear + exchange
        elif self.mode == 'iad':
            f = info + disappear + appear
        elif self.mode in ['iade', 'i2ade', 'iad2e', 'i2ad2e']:
            f = info + disappear + appear + exchange

        f = self.relu(f)
        return f
# fmt: on
