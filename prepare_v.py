import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils

C, H, W = 3, 224, 224


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   '-y',
                                   '-i', video,
                                   '-vf', "scale=224:224",
                                   '-qscale:v', "2",
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    video_list = glob.glob(os.path.join(params['video_path'], '*.avi'))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        dst = video_id
        extract_frames(video, dst)
        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        outfile = os.path.join(dir_fc, video_id.split("\\")[1] + '.npy')
        np.save(outfile, img_feats)
        shutil.rmtree(dst)


def main():
    if not os.path.exists("data/feats"):
        os.mkdir("data/feats")

    if not os.path.exists("data/feats/resnet152"):
        os.mkdir("data/feats/resnet152")

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    params = {
        "gpu": "0",
        "output_dir": "./data/feats/resnet152",
        "n_frame_steps": 40,
        "video_path": "data/video",
    }

    model = pretrainedmodels.resnet152(pretrained='imagenet')
    load_image_fn = utils.LoadTransformImage(model)

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)

    model = model.cuda()
    extract_feats(params, model, load_image_fn)


if __name__ == '__main__':
    main()
