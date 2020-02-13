#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/13
"""
import numpy as np
import os
import cv2

from root_dir import DATASET_DIR
from utils.project_utils import mkdir_if_not_exist


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def detect_img_blur(img_op):
    """
    检测图像laplacian值
    """
    gray = cv2.cvtColor(img_op, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def sigmoid_blur(val, thr=60):
    x = val - thr
    x = x / 25
    sig = 1 / (1 + np.exp(x * -1))
    return round(sig, 3)


def sigmoid_fps(val, thr=20):
    x = val - thr
    x = x / 2
    sig = 1 / (1 + np.exp(x * -1))
    return round(sig, 3)


def sigmoid_wh(w, h, thr=600):
    x = max(w, h)
    x = x - thr
    x = x / 100
    sig = 1 / (1 + np.exp(x * -1))
    return round(sig, 3)


def detect_vid_blur(vid_path):
    """
    检测视频laplacian值
    """
    print('[Info] ' + '-' * 50)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    print('[Info] 视频路径: {}, 名称: {}'.format(vid_path, vid_name))

    cap = cv2.VideoCapture(vid_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 26

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w > h:
        ratio = 1024 / w
    else:
        ratio = 1024 / h

    print('[Info] 视频尺寸: {}, 帧率: {}'.format((h, w), fps))

    gap = n_frame // 12

    # out_dir = os.path.join(DATASET_DIR, 'outs')
    # mkdir_if_not_exist(out_dir)
    # vid_dir = os.path.join(DATASET_DIR, 'outs', vid_name)
    # mkdir_if_not_exist(vid_dir)

    n_list = [k for k in range(0, n_frame, gap)]
    n_list = n_list[1:-2]

    fm_sum = 0
    for i in n_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
        # print('[Info] frame shape: {}'.format(frame.shape))

        fm = detect_img_blur(frame)

        fm_sum += fm

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img=frame,
        #             text='l: {:2f}'.format(fm),
        #             org=(10, 20),
        #             fontFace=font,
        #             fontScale=0.5,
        #             color=(0, 0, 255),
        #             thickness=2,
        #             lineType=2)
        #
        # frame_path = os.path.join(vid_dir, 'frame_{}.jpg'.format(i))
        # cv2.imwrite(frame_path, frame)

    avg = fm_sum / len(n_list)
    print('[Info] 模糊值: {}'.format(avg))

    v_blur = sigmoid_blur(avg)
    v_fps = sigmoid_fps(fps)
    v_wh = sigmoid_wh(w, h)

    return v_blur, v_fps, v_wh


def main():
    # v_path = os.path.join(DATASET_DIR, 'videos', 'positive', '303508511989705.mp4')
    # v_path = os.path.join(DATASET_DIR, 'videos', 'negative', '1026569224421716.mp4')
    v_path = os.path.join(DATASET_DIR, 'videos', 'negative', '985394418127560.mp4')
    detect_vid_blur(v_path)


if __name__ == '__main__':
    main()
