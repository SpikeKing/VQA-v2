#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/11
"""
import os
import time
import cv2
import numpy as np
import skvideo.io
import torch
from PIL import Image
from torchvision import transforms

from CNNfeatures import get_features
from VSFA import VSFA
from root_dir import MODELS_DIR, ROOT_DIR


class VideoPredictor(object):
    def __init__(self):
        self.frame_batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('[Info] device: {}'.format(self.device))
        self.model_path = os.path.join(MODELS_DIR, 'VSFA.pt')

        self.model = self.init_model()  # 初始化模型

    def init_model(self):
        """
        初始化模型
        """
        model = VSFA()
        # model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        model.eval()

        return model

    def get_feature(self, video_data):
        """
        获取视频特征
        """
        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]

        # 修改为最大边1024尺寸
        if video_width > video_height:
            ratio = 1024 / video_width
        else:
            ratio = 1024 / video_height

        width = int(video_width * ratio)
        height = int(video_height * ratio)

        transformed_video = torch.zeros([video_length, video_channel, height, width])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for frame_idx in range(video_length):
            frame = video_data[frame_idx]

            frame = Image.fromarray(frame)
            frame = frame.resize((width, height))  # 统一尺寸

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
        print('-' * 50)
        print('[Info] 视频路径: {}'.format(video_path))
        start = time.time()

        video_data = skvideo.io.vread(video_path)
        print('[Info] video shape: {}'.format(video_data.shape))

        features = self.get_feature(video_data)

        with torch.no_grad():
            input_length = features.shape[1] * torch.ones(1, 1)
            outputs = self.model(features, input_length)
            y_pred = outputs[0][0].to('cpu').numpy()

        end = time.time()

        print('[Info] Predicted quality: {}'.format(y_pred))
        print('[Info] 预测耗时: {} s'.format(end - start))
        return y_pred


def video_predictor_test():
    # video_path = os.path.join(ROOT_DIR, 'test.mp4')
    video_path = os.path.join(ROOT_DIR, 'dataset', 'videos', 'negative', '1026569224421716.mp4')
    vp = VideoPredictor()
    vp.predict_path(video_path)
    print('[Info] 视频处理完成!')


def main():
    video_predictor_test()


if __name__ == '__main__':
    main()
