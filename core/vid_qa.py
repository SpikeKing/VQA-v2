#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/11
"""
import os
import time

import cv2
import torch
from PIL import Image
from torchvision import transforms

from CNNfeatures import get_features, ResNet50
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

        self.model, self.feature_model = self.init_model()  # 初始化模型

    def init_model(self):
        """
        初始化模型
        """
        model = VSFA()
        # model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        model.eval()

        rn50_model = ResNet50()
        rn50_model.to(self.device)
        rn50_model.eval()

        return model, rn50_model

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

    def predict_vid(self, vid_path):
        """
        预测视频路径
        """
        print('[Info] 视频路径: {}'.format(vid_path))
        start = time.time()

        cap = cv2.VideoCapture(vid_path)
        vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 26

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('[Info] 视频尺寸 宽: {}, 高: {}'.format(w, h))

        print('[Info] 特征提取开始!')
        # features = self.get_feature(video_data)
        height, width = self.unify_size(h, w)  # 统一视频帧的尺寸
        print('[Info] 统一尺寸: {}, {}'.format(height, width))

        transformed_video = torch.zeros([vid_len, 3, height, width])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print('[Info] 视频长度: {}'.format(vid_len))
        for frame_idx in range(0, vid_len, 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(frame)
            frame = frame.resize((width, height))  # 统一尺寸

            frame = transform(frame)
            transformed_video[frame_idx] = frame

            if frame_idx % 200 == 0:
                print('[Info] 处理帧数: {}'.format(frame_idx))

        print('[Info] Video length: {}'.format(transformed_video.shape[0]))

        features = get_features(transformed_video, self.feature_model,
                                frame_batch_size=self.frame_batch_size,
                                device=self.device)
        features = torch.unsqueeze(features, 0)  # batch size 1
        print('[Info] 特征提取结束!')

        print('[Info] 视频预测中')
        with torch.no_grad():
            input_length = features.shape[1] * torch.ones(1, 1)
            outputs = self.model(features, input_length)
            y_pred = outputs[0][0].to('cpu').numpy()
        print('[Info] 视频预测完成!')
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
