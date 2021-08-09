from PIL import Image
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import mse_loss
import torch.optim as optim
import os
import shutil

from SIREN import *
from utils import _torch_frames2video, _pil_frames2video


def train(net, target_video, n_iters, save_path, validation_path=None):
    # preparation: clean validation directory
    if validation_path is None:
        validation_path = './validation'
    else:
        if os.path.exists(validation_path):
            shutil.rmtree(validation_path)
        os.makedirs(validation_path)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    frames_batch = []
    n_frames = int(target_video.get(cv2.CAP_PROP_FRAME_COUNT))
    h = target_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = target_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = target_video.get(cv2.CAP_PROP_FPS)
    orig_fps = 240
    orig_duration = n_frames / orig_fps
    target_fps = 30
    target_duration = 3  # seconds
    n_req_f = target_fps * target_duration
    # frame_step = int(n_frames / n_req_f)
    frame_step = int(orig_fps / target_fps)
    res = int(w // 4), int(h // 4)

    # sample frames
    # frame_from, frame_to = 0, n_frames
    # frame_from, frame_to = 72, 1128
    target_frame_length = (n_req_f * frame_step)
    frame_from = 72
    frame_to = target_frame_length + 72

    for i in range(frame_from, frame_to, frame_step):
        target_video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = target_video.read()
        if not ok:
            print('not okay: {}'.format(i))
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).convert('RGB')
        frame = frame.resize(size=res)
        if i == 0:
            frame.save('{}/GT_first_frame.png'.format(validation_path))
        # frame.save('{}/frame_{}.png'.format(validation_path, i))
        frames_batch.append(to_tensor(frame))
        # frames_batch.append(frame)
    # _pil_frames2video(frames_batch, './{}/fps30.mp4'.format(validation_path), fps=target_fps)

    frames_batch = torch.stack(frames_batch).cuda()
    total_frames, _, h, w = frames_batch.shape

    print('start training...')
    for i in range(n_iters):
        coords = uniform_coordinates_3d(h, w, total_frames).cuda()
        output = net(coords)
        output = flat_to_3d(output, (h, w, total_frames))
        loss = mse_loss(output, frames_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2500 == 0:
            print(i + 1, loss.item())

            to_pil_image(output[0].cpu()).save('{}/at_{}.png'.format(validation_path, i + 1))
            torch.save(net.module.state_dict(), save_path)

    # training finished.
    torch.save(net.module.state_dict(), save_path)

    # final validation
    with torch.no_grad():
        coords = uniform_coordinates_3d(h, w, total_frames).cuda()
        output = net(coords)
        output = flat_to_3d(output, (h, w, total_frames))

    # write video file
    _torch_frames2video(output, './{}/final_video.mp4'.format(validation_path), target_fps)

    # codec = cv2.VideoWriter_fourcc(*'mp4v')
    # vout = cv2.VideoWriter('./{}/final_video.mp4'.format(validation_path), codec, target_fps, (w, h))
    # for i in range(total_frames):
    #     new_frame = cv2.cvtColor(np.array(to_pil_image(output[i].cpu())), cv2.COLOR_RGB2BGR)
    #     vout.write(new_frame)
    # vout.release()

    target_video.release()


def inference(net, size):
    h, w, t = size
    with torch.no_grad():
        coords = uniform_coordinates_3d(h, w, t).cuda()
        coords = coords * torch.tensor([30., 30., 1.]).cuda()
        n = len(coords)
        b = 100000
        outputs = []
        for i in range(0, n, b):
            if i + b < n:
                part_coords = coords[i: i + b]
            else:
                part_coords = coords[i: n]
            output = net(part_coords)
            # output = flat_to_3d(output, (h, w, t))
            outputs.append(output.cpu())
        output = torch.cat(outputs, dim=0)
        output = flat_to_3d(output, (h, w, t))
    _torch_frames2video(output, './test.mp4', 30)


if __name__ == '__main__':
    is_train = False
    net = SirenNet(dim_in=3, dim_hidden=256, dim_out=3, n_layers=5)
    net = nn.DataParallel(net)
    video = cv2.VideoCapture('./examples/IMG_0153.m4v')
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    n_iters = 25000
    save_path = './30fps.pth'
    validation_path = './validation_30fps'

    if is_train:
        train(net.cuda(), video, n_iters, save_path, validation_path)

    else:
        net.module.load_state_dict(torch.load(save_path))
        inference(net.cuda(), size=(h//4, w//4, 120))
