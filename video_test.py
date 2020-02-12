#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/11
"""

import os

import xlsxwriter

from img_plugin.predictor_with_img import PredictorWithImage
from predictor import VideoPredictor
from root_dir import DATASET_DIR
from utils.project_utils import traverse_dir_files, mkdir_if_not_exist


def video_test():
    """
    视频测试
    """
    vid_p_dir = os.path.join(DATASET_DIR, 'videos', 'positive')
    vid_n_dir = os.path.join(DATASET_DIR, 'videos', 'negative')

    paths_p_list, names_p_list = traverse_dir_files(vid_p_dir)
    paths_n_list, names_n_list = traverse_dir_files(vid_n_dir)

    names_list = names_p_list + names_n_list
    paths_list = paths_p_list + paths_n_list

    vp = VideoPredictor()
    pwi = PredictorWithImage()

    out_dir = os.path.join(DATASET_DIR, 'outs')
    mkdir_if_not_exist(out_dir)
    out_excel_file = os.path.join(out_dir, 'res.xlsx')
    print('[Info] 视频结果文件: {}'.format(out_excel_file))

    # add_sheet is used to create sheet.
    workbook = xlsxwriter.Workbook(out_excel_file)
    worksheet = workbook.add_worksheet()

    row = 0

    print('[Info] 视频总数: {}'.format(len(names_list)))
    worksheet.write(row, 0, u'视频名称')
    worksheet.write(row, 1, u'视频评分')
    worksheet.write(row, 2, u'图像质量')
    worksheet.write(row, 3, u'图像美学')
    row += 1

    count = 0
    for name, path in zip(names_list, paths_list):
        try:
            score = vp.predict_path(path)
            score_tech, score_aest = pwi.predict_video(path)
            print('[Info] 视频: {}, 视频评分: {}, 图像质量: {}, 图像美学: {}'.format(name, score, score_tech, score_aest))

            worksheet.write(row, 0, name)
            worksheet.write(row, 1, score)
            worksheet.write(row, 2, score_tech)
            worksheet.write(row, 3, score_aest)
            row += 1

            count += 1
            print('[Info] 已处理视频: {} / {}'.format(count, len(names_list)))
        except Exception as e:
            print(e)
            print('[Info] 错误视频: {}'.format(name))

    workbook.close()

    print('[Info] 视频处理全部完成!')


def main():
    video_test()


if __name__ == '__main__':
    main()
