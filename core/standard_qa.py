#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/13
"""
import os

import cv2

from root_dir import DATASET_DIR
from utils.vpa_utils import sigmoid_thr


class StandardQualityAssessment(object):
    """
    视频标准值的质量评估
    """

    def __init__(self):
        self.n_frames = 10  # 检测帧数

    @staticmethod
    def variance_of_laplacian(img_np):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(img_np, cv2.CV_64F).var()

    def get_img_blur(self, img_op):
        """
        检测图像laplacian值
        """
        gray = cv2.cvtColor(img_op, cv2.COLOR_BGR2GRAY)
        vol = self.variance_of_laplacian(gray)

        return vol

    def norm_blur(self, v_blur):
        """
        归一化模糊
        """
        # 归一化返回0~1的值，范围0~210转换为0~1
        s = sigmoid_thr(v_blur, 60, 30)

        return s

    def norm_fps(self, fps):
        """
        归一化FPS
        """
        # 归一化返回0~1的值，范围10~30转换为0~1
        s = sigmoid_thr(fps, 20, 2)

        return s

    def norm_size(self, ms):
        """
        归一化最大尺寸
        """
        # 归一化返回0~1的值，范围0~1200转换为0~1
        s = sigmoid_thr(ms, 600, 120)

        return s

    def predict_video(self, vid_path):
        """
        视频质量值
        """
        cap = cv2.VideoCapture(vid_path)
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 26

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        gap = n_frame // self.n_frames
        n_list = [k for k in range(0, n_frame, gap)]
        n_list = n_list[1:-2]  # 去掉前后两帧

        ratio = float(1024 / max(w, h))

        sum_b = 0  # 视频的blur总值

        for i in n_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
            vb = self.get_img_blur(frame)

            sum_b += vb

        avg = sum_b / self.n_frames
        print('[Info] 视频模糊值: {}'.format(avg))

        nb = self.norm_blur(avg)
        nf = self.norm_fps(fps)
        nz = self.norm_size(max(w, h))

        # 模糊值、FPS值、尺寸值
        return nb, nf, nz


def main():
    v_path = os.path.join(DATASET_DIR, 'videos', 'negative', '985394418127560.mp4')

    sqa = StandardQualityAssessment()
    nb, nf, nz = sqa.predict_video(v_path)

    print('[Info] 模糊值: {}, FPS值: {}, 尺寸值: {}'.format(nb, nf, nz))


if __name__ == '__main__':
    main()
