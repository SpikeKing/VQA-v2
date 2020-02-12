#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/6
"""

import os

import cv2
import numpy as np
from PIL import Image as Image

from img_plugin.core.handlers.model_builder import Nima
from img_plugin.core.utils.utils import calc_mean_score

from root_dir import DATA_DIR, MODELS_DIR, DATASET_DIR
from utils.vpa_utils import norm_img


class ImgPredictor(object):

    def __init__(self):
        self.base_model_name = 'MobileNet'
        self.weights_tech = os.path.join(MODELS_DIR, 'MobileNet', 'weights_mobilenet_technical_0.11.hdf5')
        self.weights_aest = os.path.join(MODELS_DIR, 'MobileNet', 'weights_mobilenet_aesthetic_0.07.hdf5')

        self.model_tech = self.load_model(self.weights_tech)
        self.model_aest = self.load_model(self.weights_aest)

    @staticmethod
    def resize_img(img_pil):
        """
        Resize图像
        """
        img = img_pil.resize((224, 224), Image.NEAREST)
        img_np = np.asarray(img)
        return img_np

    @staticmethod
    def crop_img(img_pil):
        """
        裁剪中间图像
        """
        std_shape = (320, 320)
        img = img_pil.resize(std_shape, Image.NEAREST)
        img_np = np.asarray(img)
        x = (std_shape[0] - 224) // 2
        img_np = img_np[x:x + 224, x:x + 224]
        return img_np

    def load_model(self, weights_path):
        """
        加载模型
        """
        nima = Nima(self.base_model_name, weights=None)
        nima.build()
        nima.nima_model.load_weights(weights_path)  # 加载参数

        return nima.nima_model

    def predict_img_path(self, img_path):
        img_pil = Image.open(img_path)

        score_tech, score_aest = self.predict_img(img_pil)
        return score_tech, score_aest

    def predict_img_op(self, img_op):
        img = cv2.cvtColor(img_op, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        score_tech, score_aest = self.predict_img(img_pil)
        return score_tech, score_aest

    def predict_img(self, img_pil):
        """
        预测图像，Pillow格式
        """
        img_tech = self.crop_img(img_pil)
        img_aest = self.resize_img(img_pil)  # 与标准相同

        img_tech_n = norm_img(img_tech)
        img_tech_n = img_tech_n.astype(np.float32)
        img_tech_b = np.expand_dims(img_tech_n, axis=0)

        img_aest_n = norm_img(img_aest)
        img_aest_n = img_aest_n.astype(np.float32)
        img_aest_b = np.expand_dims(img_aest_n, axis=0)

        score_tech_list = self.model_tech.predict(img_tech_b)
        score_aest_list = self.model_aest.predict(img_aest_b)

        score_tech = calc_mean_score(score_tech_list)
        # print('[Info] 视频质量: {}'.format(score_tech))

        score_aest = calc_mean_score(score_aest_list)
        # print('[Info] 视频美学: {}'.format(score_aest))

        # 图像质量
        #
        # img_tech = cv2.cvtColor(np.asarray(img_tech), cv2.COLOR_RGB2BGR)
        # frame_tech_path = os.path.join(DATASET_DIR, 'outs', 'frame_tech.jpg')
        # self.save_img(frame_tech_path, img_tech, score_tech)
        #
        # img_aest = cv2.cvtColor(np.asarray(img_aest), cv2.COLOR_RGB2BGR)
        # frame_aest_path = os.path.join(DATASET_DIR, 'outs', 'frame_aest.jpg')
        # self.save_img(frame_aest_path, img_aest, score_aest)

        return score_tech, score_aest

    def save_img(self, frame_path, frame, score):
        """
        存储图像
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(img=frame,
                    text='score: {:2f}'.format(score),
                    org=(10, 20),
                    fontFace=font,
                    fontScale=0.5,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=2)

        cv2.imwrite(frame_path, frame)


def img_predictor_test():
    """
    ImgPredictor测试
    """
    img_test_path = os.path.join(DATASET_DIR, 'imgs', 'landscape.jpg')
    print('[Info] 图像路径: {}'.format(img_test_path))

    ip = ImgPredictor()
    score_tech, score_aest = ip.predict_img_path(img_test_path)
    print('[Info] 图像质量: {}, 图像美学: {}'.format(score_tech, score_aest))


def main():
    img_predictor_test()


if __name__ == '__main__':
    main()
