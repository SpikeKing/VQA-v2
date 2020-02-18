#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/11
"""
import os
import time

import skvideo.io
import torch
from PIL import Image
from torchvision import transforms

from CNNfeatures import get_features
from VSFA import VSFA
from root_dir import MODELS_DIR, ROOT_DIR


class VideoQualityAssessment(object):
    """
    视频整体的质量评估
    """

    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, 'VSFA.pt')  # 模型路径

        self.frame_batch_size = 32  # batch_size
        self.std_size = 1024  # 视频图像尺寸

        # CPU和GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('[Info] device: {}'.format(self.device))

        self.model = self.init_model()  # 初始化模型

    def init_model(self):
        """
        初始化模型
        """
        model = VSFA()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        # model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        model.eval()

        return model

    def unify_size(self, video_height, video_width):
        """
        统一最长边的尺寸
        """
        # 最长边修改为标准尺寸
        if video_width > video_height:
            ratio = self.std_size / video_width
        else:
            ratio = self.std_size / video_height

        height = int(video_height * ratio)
        width = int(video_width * ratio)

        return height, width

    def get_feature(self, video_data):
        """
        获取视频特征
        """
        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]

        height, width = self.unify_size(video_height, video_width)  # 统一视频帧的尺寸
        print('[Info] 统一尺寸: {}, {}'.format(height, width))

        transformed_video = torch.zeros([video_length, video_channel, height, width])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print('[Info] 视频长度: {}'.format(video_length))
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]

            frame = Image.fromarray(frame)
            frame = frame.resize((width, height))  # 统一尺寸

            frame = transform(frame)
            transformed_video[frame_idx] = frame

            if frame_idx % 100 == 0:
                print('[Info] 处理帧数: {}'.format(frame_idx))

        print('[Info] Video length: {}'.format(transformed_video.shape[0]))

        features = get_features(transformed_video, frame_batch_size=self.frame_batch_size, device=self.device)
        features = torch.unsqueeze(features, 0)  # batch size 1

        return features

    def predict_vid(self, video_path):
        """
        预测视频路径
        """
        print('[Info] 视频路径: {}'.format(video_path))
        start = time.time()

        video_data = skvideo.io.vread(video_path)
        print('[Info] 视频尺寸: {}'.format(video_data.shape))

        features = self.get_feature(video_data)

        with torch.no_grad():
            input_length = features.shape[1] * torch.ones(1, 1)
            outputs = self.model(features, input_length)
            y_pred = outputs[0][0].to('cpu').numpy()

        end = time.time()

        print('[Info] 预测的视频质量: {}'.format(y_pred))
        print('[Info] 预测耗时: {} s'.format(end - start))
        return y_pred


def video_predictor_test():
    # video_path = os.path.join(ROOT_DIR, 'test.mp4')
    vid_path = os.path.join(ROOT_DIR, 'dataset', 'videos', 'negative', '1026569224421716.mp4')
    vqa = VideoQualityAssessment()
    vqa.predict_vid(vid_path)
    print('[Info] 视频处理完成!')


def main():
    video_predictor_test()


if __name__ == '__main__':
    main()
