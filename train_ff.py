from PIL import Image
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import mse_loss
import torch.optim as optim
import os
import shutil

from SIREN import *


def train(net, target_img, fourier_feats, n_iters, save_path, validation_path=None):
    # preparation: clean validation directory
    if validation_path is None:
        validation_path = './validation'
    else:
        if os.path.exists(validation_path):
            shutil.rmtree(validation_path)
        os.makedirs(validation_path)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    img = to_tensor(target_img).unsqueeze(0).cuda()
    h, w = img.shape[2:]

    print('start training...')
    for i in range(n_iters):
        coords = uniform_coordinates(h, w).cuda()
        coords = fourier_feats.map(coords)

        output = net(coords)
        output = flat_to_2d(output, (h, w))
        loss = mse_loss(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2500 == 0:
            print(i + 1, loss.item())

            to_pil_image(output[0].cpu()).save('{}/at_{}.png'.format(validation_path, i + 1))
            torch.save(net.state_dict(), save_path)
            torch.save(fourier_feats.B.cpu(), save_path[:-4] + '_FF.pth')

    # training finished.
    torch.save(net.state_dict(), save_path)
    torch.save(fourier_feats.B.cpu(), save_path[:-4] + '_FF.pth')

    # final validation
    with torch.no_grad():
        coords = uniform_coordinates(h, w).cuda()
        coords = fourier_feats.map(coords)

        output = net(coords)
        output = flat_to_2d(output, (h, w))

    to_pil_image(output[0].cpu()).save('./for_fun.png')


def inference(net, fourier_feats, size):
    h, w = size
    with torch.no_grad():
        coords = uniform_coordinates(h, w).cuda()
        coords = fourier_feats.map(coords)
        output = net(coords, (h, w))
        output = flat_to_2d(output, (h, w))

    to_pil_image(output[0].cpu()).save('./for_fun.png')


if __name__ == '__main__':
    is_train = True
    # net = SirenNet(dim_in=2, dim_hidden=256, dim_out=3, n_layers=5)
    ff_dim = 256
    net = MLP(dim_in=2, dim_hidden=256, dim_out=3, n_layers=5, fourier_dim=ff_dim)
    fourier_feats = InputMapping(mapping_size=ff_dim, dim=2)
    fourier_feats.B = fourier_feats.B.cuda()

    img = Image.open('./examples/0002x4.png')
    w, h = img.size
    n_iters = 25000
    save_path = './ff_for_fun.pth'
    validation_path = './ff_validation'

    if is_train:
        train(net.cuda(), img, fourier_feats, n_iters, save_path, validation_path)

    else:
        net.load_state_dict(torch.load(save_path))
        # net.module.load(save_path)
        fourier_feats.B = torch.load(save_path[:-4] + '_FF.pth').cuda()
        inference(net.cuda(), fourier_feats, size=(h, w))

