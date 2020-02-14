#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/14
"""
import os

from core.standard_qa import StandardQualityAssessment
from core.video_qa import VideoQualityAssessment
from root_dir import ROOT_DIR


class VideoPredictor(object):
    """
    视频预测
    """

    def __init__(self):
        self.sqa = StandardQualityAssessment()
        self.vqa = VideoQualityAssessment()

    def predict_video(self, vid_path):
        nb, nf, nz = self.sqa.predict_video(vid_path)
        vq = self.vqa.predict_path(vid_path)

        # 最终得分
        final_val = vq * 0.85 + nb * 0.05 + nf * 0.05 + nz * 0.05
        return final_val


def main():
    vid_path = os.path.join(ROOT_DIR, 'dataset', 'videos', 'negative', '1026569224421716.mp4')

    vp = VideoPredictor()
    fv = vp.predict_video(vid_path)
    print('[Info] 最终得分: {}'.format(fv))


if __name__ == '__main__':
    main()
