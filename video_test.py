#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/11
"""

import os

import xlsxwriter

from img_plugins.img_blur import detect_vid_blur
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

    p_len, n_len = len(names_p_list), len(names_n_list)

    names_list = names_p_list + names_n_list
    paths_list = paths_p_list + paths_n_list

    vp = VideoPredictor()

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
    worksheet.write(row, 2, u'视频模糊')
    worksheet.write(row, 3, u'FPS')
    worksheet.write(row, 4, u'尺寸')
    worksheet.write(row, 5, u'最终得分')
    worksheet.write(row, 6, u'正负示例')
    row += 1

    count = 0
    for name, path in zip(names_list, paths_list):
        try:
            score = vp.predict_path(path)
            print('[Info] 视频: {}, 视频评分: {}'.format(name, score))

            v_blur, v_fps, v_wh = detect_vid_blur(path)

            v_final = score * 0.50 + v_blur * 0.10 + v_fps * 0.15 + v_wh * 0.15  # 最终评分

            worksheet.write(row, 0, name)
            worksheet.write(row, 1, score)
            worksheet.write(row, 2, v_blur)
            worksheet.write(row, 3, v_fps)
            worksheet.write(row, 4, v_wh)
            worksheet.write(row, 5, v_final)

            if count < p_len:
                worksheet.write(row, 6, "p")
            else:
                worksheet.write(row, 6, "n")

            row += 1

        except Exception as e:
            print(e)
            print('[Info] 错误视频: {}'.format(name))

        count += 1
        print('[Info] 已处理视频: {} / {}'.format(count, len(names_list)))

    workbook.close()

    print('[Info] 视频处理全部完成!')


def main():
    video_test()


if __name__ == '__main__':
    main()
