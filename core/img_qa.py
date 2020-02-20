#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/18
"""
import os

import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from core.img_core.model import IQAModel
from root_dir import DATASET_DIR, MODELS_DIR


@DeprecationWarning
class ImgQualityAssessment(object):
    """
    图像质量评估，目前效果较差
    """

    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, 'epoch-57.pkl')
        self.model, self.device = self.init_model()
        self.test_transform = self.get_test_transform()

        self.n_frames = 10  # 检测视频的图像数

    def init_model(self):
        base_model = models.vgg16(pretrained=True)
        model = IQAModel(base_model)
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(self.model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        return model, device

    def get_test_transform(self):
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        return test_transform

    def predict_img(self, img_pil):
        """
        预测图像
        """
        imt = self.test_transform(img_pil)

        imt = imt.unsqueeze(dim=0)
        imt = imt.to(self.device)
        with torch.no_grad():
            out = self.model(imt)
        out = out.view(10, 1)
        mean, std = 0.0, 0.0
        for j, e in enumerate(out, 1):
            mean += j * e
        for k, e in enumerate(out, 1):
            std += (e * (k - mean) ** 2) ** 0.5

        mean, std = float(mean), float(std)
        return mean, std

    def predict_img_path(self, img_path):
        """
        预测图像路径
        """
        imt = Image.open(img_path)
        mean, std = self.predict_img(imt)

        return mean, std

    def predict_vid(self, vid_path):
        """
        基于视频帧计算的视频质量值
        """
        cap = cv2.VideoCapture(vid_path)
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        gap = n_frame // self.n_frames
        n_list = [k for k in range(0, n_frame, gap)]
        n_list = n_list[1:-2]  # 去掉前后两帧

        sum_mean, sum_std = 0.0, 0.0  # 视频的blur总值

        for i in n_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mean, std = self.predict_img(img_pil)

            sum_mean += mean
            sum_std += std

        avg_mean = sum_mean / self.n_frames
        avg_std = sum_std / self.n_frames

        print('[Info] 视频质量均值: {}, 方差: {}'.format(avg_mean, avg_std))

        norm_mean = avg_mean / 10.0
        norm_std = avg_std / 10.0

        return norm_mean, norm_std


def main():
    iqa = ImgQualityAssessment()

    img_path = os.path.join(DATASET_DIR, 'imgs', 'landscape.jpg')
    print('[Info] 图像路径: {}'.format(img_path))
    mean, std = iqa.predict_img_path(img_path)
    print('[Info] 均值: {}, 方差: {}'.format(mean, std))


if __name__ == '__main__':
    main()
