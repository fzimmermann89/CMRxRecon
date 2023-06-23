import torch
import torch.nn as nn


class ClipAct(nn.Module):
    def __init__(self):
        super(ClipAct, self).__init__()

    def forward(self, x, threshold):
        is_complex = x.is_complex()
        if is_complex:
            x = torch.view_as_real(x)
            threshold = threshold.unsqueeze(-1)

        x = torch.clamp(x, -threshold, threshold)

        if is_complex:
            x = torch.view_as_complex(x)

        return x


class SoftShrinkAct_(nn.Module):
    def __init__(self):
        super(SoftShrinkAct_, self).__init__()

    def forward(self, x, threshold):
        is_complex = x.is_complex()
        if is_complex:
            x = torch.view_as_real(x)
            threshold = threshold.unsqueeze(-1)

        x = torch.nn.functional.softshrink(x, threshold)

        if is_complex:
            x = torch.view_as_complex(x)

        return x
