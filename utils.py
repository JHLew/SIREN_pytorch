import cv2
from PIL import Image
import numpy as np
import os
import shutil


def _torch_frames2video(frames, output_path, fps=30):
    # write video file
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    if not '.mp4' in output_path:
        raise NameError('Incorrect output file name. Must be .mp4 format.')

    total_frames, _, h, w = frames.shape
    vout = cv2.VideoWriter(output_path, codec, fps, (w, h))
    for i in range(total_frames):
        new_frame = cv2.cvtColor(np.array(to_pil_image(frames[i].cpu())), cv2.COLOR_RGB2BGR)
        vout.write(new_frame)
    vout.release()


def _pil_frames2video(frames, output_path, fps=30):
    # write video file
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    if not '.mp4' in output_path:
        raise NameError('Incorrect output file name. Must be .mp4 format.')

    total_frames = len(frames)
    w, h = frames[0].size
    vout = cv2.VideoWriter(output_path, codec, fps, (w, h))
    for i in range(total_frames):
        new_frame = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_RGB2BGR)
        vout.write(new_frame)
    vout.release()


def video2frames(video_path, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    for i in range(n_frames):
        # video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = video.read()
        if not ok:
            raise IOError(f'Error in reading video: frame no. {i}')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).convert('RGB')
        frame.save(os.path.join(output_path, f'frame_{i}.png'))

    video.release()


def reduce_fps(video_path, output_path, _from=240, _to=30):
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(output_path, codec, _to, (w, h))
    rate = int(_from / _to)

    frames = []
    for i in range(0, n_frames, rate):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = video.read()
        if not ok:
            raise IOError(f'Error in reading video: frame no. {i}')
        vout.write(frame)
    video.release()
    vout.release()


if __name__ == '__main__':
    # video2frames('./examples/Stop motion animation fruit and vegetables.mp4', './vegetables')
    reduce_fps('./examples/IMG_0153.m4v', './IMG_0153_30fps.mp4')