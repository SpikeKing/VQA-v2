import cv2
import os
import json
import keras
import numpy as np


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def random_crop(img, crop_dims):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch, 'image height is less than crop height'
    assert w >= cw, 'image width is less than crop width'
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y:(y + ch), x:(x + cw), :]


def random_horizontal_flip(img):
    assert len(img.shape) == 3, 'input tensor must have 3 dimensions (height, width, channels)'
    assert img.shape[2] == 3, 'image not in channels last format'
    if np.random.random() < 0.5:
        img = img.swapaxes(1, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 1)
    return img


def load_image(img_file, target_size):
    return np.asarray(keras.preprocessing.image.load_img(img_file, target_size=target_size))
    # img_np = cv2.imread(img_file)
    # print('[Info] max: {}, min: {}, avg: {}'.format(np.min(img_np), np.max(img_np), np.mean(img_np)))
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # print('[Info] max: {}, min: {}, avg: {}'.format(np.min(img_np), np.max(img_np), np.mean(img_np)))
    # img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_NEAREST)
    # print('[Info] max: {}, min: {}, avg: {}'.format(np.min(img_np), np.max(img_np), np.mean(img_np)))
    # return img_np


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
