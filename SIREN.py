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
# class InputMapping(nn.Module):
    def __init__(self, mapping_size=256, dim=2, B='gaussian', sigma=10):
        # super(InputMapping, self).__init__()
        if B == 'gaussian':
            self.B = torch.randn((dim, mapping_size)) * sigma
        elif B == 'uniform':
            self.B = torch.rand((dim, mapping_size)) * sigma
        else:
            raise ValueError('wrong B type. Got {}'.format(B))
        # self.B = nn.Parameter(self.B)

    def map(self, x):
    # def forward(self, x):
        x_proj = torch.mm((2 * np.pi * x), self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def uniform_coordinates(h, w, _range=(-1, 1), flatten=True):
    _from, _to = _range
    coords = [torch.linspace(_from, _to, steps=h), torch.linspace(_from, _to, steps=w)]
    mgrid = torch.stack(torch.meshgrid(*coords), dim=-1)
    if flatten:
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')

    return mgrid.detach()


def uniform_coordinates_3d(h, w, t, _range=(-1, 1), flatten=True):
    _from, _to = _range
    coords = [torch.linspace(_from, _to, steps=h),
              torch.linspace(_from, _to, steps=w),
              torch.linspace(_from, _to, steps=t)]
    mgrid = torch.stack(torch.meshgrid(*coords), dim=-1)
    if flatten:
        mgrid = rearrange(mgrid, 'h w t c -> (h w t) c')

    return mgrid.detach()


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False):
        super(Siren, self).__init__()
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
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers, w0=1., w0_initial=30., c=6.):
        super(SirenNet, self).__init__()
        self.num_layers = n_layers
        self.dim_hidden = dim_hidden

        layers = []
        for i in range(self.num_layers):
            if i == self.num_layers - 1:  # if final layer
                final_layer = nn.Linear(dim_hidden, dim_out)
                w_std = np.sqrt(c / dim_hidden) / w0
                final_layer.weight.data.uniform_(-w_std, w_std)
                layers.append(final_layer)
                break

            if i == 0:  # if first layer
                layer_w0 = w0_initial
                layer_dim_in = dim_in
                is_first = True
            else:
                layer_w0 = w0
                layer_dim_in = dim_hidden
                is_first = False

            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                c=c,
                is_first=is_first,
            ))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, out_size=None):
        out = self.layers(x)
        if out_size is not None:
            if len(out_size) == 2:
                h, w = out_size
                out = rearrange(out, '(h w) c -> () c h w', h=h, w=w)
            elif len(out_size) == 3:
                h, w, t = out_size
                out = rearrange(out, '(h w t) c -> t c h w', h=h, w=w, t=t)
            else:
                raise ValueError('wrong output size')

        return postprocess(out)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers, fourier_dim=256):
        super(MLP, self).__init__()
        self.num_layers = n_layers
        if fourier_dim is None:
            self.use_fourier = False
        else:
            self.use_fourier = True
        if self.use_fourier:
            dim_in = fourier_dim * 2
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(dim_in, dim_hidden))
                layers.append(nn.ReLU())
            elif i == self.num_layers - 1:
                layers.append(nn.Linear(dim_hidden, dim_out))
            else:
                layers.append(nn.Linear(dim_hidden, dim_hidden))
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x, out_size=None):
        out = self.layers(x)

        if out_size is not None:
            if len(out_size) == 2:
                h, w = out_size
                out = rearrange(out, '(h w) c -> () c h w', h=h, w=w)
            elif len(out_size) == 3:
                h, w, t = out_size
                out = rearrange(out, '(h w t) c -> t c h w', h=h, w=w, t=t)
            else:
                raise ValueError('wrong output size')

        return postprocess(out)

    # def save(self, path):
    #     state_dict = {'MLP': self.state_dict()}
    #     if self.fourier_feats is not None:
    #             state_dict['FF'] = self.fourier_feats.B
    #     torch.save(state_dict, path)
    #
    # def load(self, path):
    #     state_dict = torch.load(path)
    #     self.load_state_dict(state_dict['MLP'])
    #     if len(state_dict) == 2 and self.fourier_feats is not None:
    #         self.fourier_feats.B = state_dict['FF']


def flat_to_2d(_in, _size):
    h, w = _size
    out = rearrange(_in, '(h w) c -> () c h w', h=h, w=w)
    return out


def flat_to_3d(_in, _size):
    h, w, t = _size
    out = rearrange(_in, '(h w t) c -> t c h w', h=h, w=w, t=t)
    return out

