from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import mse_loss
import torch.optim as optim
import os
import shutil

from SIREN import *


def train(net, target_img, n_iters, save_path, validation_path=None):
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
        output = net(coords, (h, w))
        loss = mse_loss(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2500 == 0:
            print(i + 1, loss.item())

            with torch.no_grad():
                coords = uniform_coordinates(h, w).cuda()
                output = net(coords, (h, w))

            to_pil_image(output[0].cpu()).save('{}/{}.png'.format(validation_path, i + 1))
            torch.save(net.state_dict(), save_path)

    coords = uniform_coordinates(h, w).cuda()
    output = net(coords, (h, w))
    to_pil_image(output[0].cpu()).save('{}/final.png'.format(validation_path))

    torch.save(net.state_dict(), save_path)


def inference(net, size):
    h, w = size
    with torch.no_grad():
        coords = uniform_coordinates(h, w).cuda()
        output = net(coords, (h, w))

    to_pil_image(output[0].cpu()).save('./for_fun.png')


if __name__ == '__main__':
    is_train = True
    # net = SirenNet(dim_in=2, dim_hidden=256, dim_out=3, n_hidden_layers=5)
    img = Image.open('./examples/0002x4.png')
    w, h = img.size
    n_iters = 25000
    save_path = './for_fun.pth'
    validation_path = './validation_for_fun'

    if is_train:
        train(net.cuda(), img, n_iters, save_path, validation_path)

    else:
        net.load_state_dict(torch.load(save_path))
        inference(net.cuda(), size=(h, w))



