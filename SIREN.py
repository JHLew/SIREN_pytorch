import torch
from torch import nn
from einops import rearrange
import numpy as np


# normalize (0, 1) to (-1, 1)
def preprocess(t):
    return t * 2 - 1

# denormalize from (-1, 1) to (0, 1)
def postprocess(t):
    return torch.clamp((t + 1) / 2, min=0, max=1)


class InputMapping:
    def __init__(self, mapping_size=256, B='gaussian', sigma=10):
        if B == 'gaussian':
            self.B = torch.randn((mapping_size, 2), requires_grad=True) * sigma
        else:
            self.B = None

    def map(self, x):
        if self.B is None:
            return x
        else:
            x_proj = torch.bmm((2 * np.pi * x), self.B.T)
            return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


def uniform_coordinates(h, w, _range=(-1, 1), flatten=True):
    _from, _to = _range(-1, 1)
    coords = [torch.linspace(_from, _to, steps=h), torch.linspace(_from, _to, steps=w)]
    mgrid = torch.stack(torch.meshgrid(*coords), dim=-1)
    if flatten:
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        # mgrid = mgrid.view(h, w, 2)

    return mgrid.detach()


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first

        self.init_wb(c=c, w0=w0)
        self.w0 = w0
        self.c = c
        self.is_first = is_first

    def init_wb(self, c, w0):
        self.fc = nn.Linear(self.dim_in, self.dim_out)

        w_std = 1 / self.dim_in if self.is_first else np.sqrt(c / self.dim_in) / w0

        self.fc.weight.data.uniform_(-w_std, w_std)
        self.fc.bias.data.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.fc(x)
        if self.is_first:
            out = self.w0 * out
        out = torch.sin(out)
        return out


class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_hidden_layers, w0=1., w0_initial=30., c=6.):
        super().__init__()
        self.num_layers = n_hidden_layers
        self.dim_hidden = dim_hidden

        layers = []
        is_first = True
        for _ in range(self.num_layers):
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                c=c,
                is_first=is_first,
            ))

            if is_first:
                is_first = False

        final_layer = nn.Linear(dim_hidden, dim_out)
        w_std = np.sqrt(c / dim_hidden) / w0
        final_layer.weight.data.uniform_(-w_std, w_std)
        layers.append(final_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x=None, out_size=None):
        out = self.layers(x)
        if out_size is not None:
            h, w = out_size
            out = rearrange(out, '(h w) c -> () c h w', h=h, w=w)
            out = out.permute(1, 0).view(1, 3, h, w)


        return postprocess(out)

