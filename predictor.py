#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/11
"""
import os
import torch
from torchvision import transforms
import skvideo.io
from PIL import Image
import numpy as np
from VSFA import VSFA
from CNNfeatures import get_features
from argparse import ArgumentParser
import time

from root_dir import MODELS_DIR, ROOT_DIR


class VideoPredictor(object):
    def __init__(self):
        self.frame_batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(MODELS_DIR, 'VSFA.pt')

        self.model = self.init_model()

    def init_model(self):
        """
        初始化模型
        """
        model = VSFA()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.to(self.device)
        model.eval()

        return model

    def get_feature(self, video_data):
        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]

        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        print('Video length: {}'.format(transformed_video.shape[0]))

        features = get_features(transformed_video, frame_batch_size=self.frame_batch_size, device=self.device)
        features = torch.unsqueeze(features, 0)  # batch size 1
        return features

    def predict_path(self, video_path):
        """
        预测视频路径
        """
        start = time.time()

        video_data = skvideo.io.vread(video_path)

        features = self.get_feature(video_data)

        with torch.no_grad():
            input_length = features.shape[1] * torch.ones(1, 1)
            outputs = self.model(features, input_length)
            y_pred = outputs[0][0].to('cpu').numpy()
            print("Predicted quality: {}".format(y_pred))

        end = time.time()
        print('Time: {} s'.format(end - start))


def video_predictor_test():
    video_path = os.path.join(ROOT_DIR, 'test.mp4')
    vp = VideoPredictor()
    vp.predict_path(video_path)


def main():
    video_predictor_test()


if __name__ == '__main__':
    main()
