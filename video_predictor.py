#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/14
"""
import os

from core.std_qa import StandardQualityAssessment
from core.vid_qa import VideoQualityAssessment
from root_dir import ROOT_DIR


class VideoPredictor(object):
    """
    视频预测
    """

    def __init__(self):
        self.sqa = StandardQualityAssessment()
        self.vqa = VideoQualityAssessment()
        # self.iqa = ImgQualityAssessment()

    def predict_vid(self, vid_path):
        final_val, _, _, _, _ = self.predict_vid_detail(vid_path)
        return final_val

    def predict_vid_detail(self, vid_path):
        vq = self.vqa.predict_vid(vid_path)
        nb, nf, nz = self.sqa.predict_vid(vid_path)

        # nm, ns = self.iqa.predict_vid(vid_path)

        # 最终得分
        final_val = vq * 0.80 + nb * 0.05 + nf * 0.10 + nz * 0.05
        return final_val, vq, nb, nf, nz


def main():
    vid_path = os.path.join(ROOT_DIR, 'dataset', 'videos', 'negative', '1026569224421716.mp4')

    vp = VideoPredictor()
    fv = vp.predict_vid(vid_path)
    print('[Info] 最终得分: {}'.format(fv))


if __name__ == '__main__':
    main()
