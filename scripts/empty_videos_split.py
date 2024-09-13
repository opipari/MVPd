import os
import json

from typing import List, Optional, Tuple, Callable, Union

import cv2
import numpy as np
from PIL import Image

from panopticapi.utils import rgb2id



def get_label(
    path
) -> np.ndarray:

    label = Image.open(path)
    label = np.array(label, dtype=np.uint8)
    label = rgb2id(label)

    return label



def reject_empty_videos(data, root, area_threshold=0.5):

    accepted_videos = []
    rejected_videos = []

    label_root = os.path.join(root, 'panomasksRGB')

    for i, video_meta in enumerate(data['annotations']):

        has_empty = False
        for anno_meta in video_meta['annotations']:
            file_path = os.path.join(label_root, video_meta["video_name"], anno_meta["file_name"])
            label = get_label(file_path)

            area = label.shape[0]*label.shape[1]

            if (label==0).sum() > (area*area_threshold):
                has_empty = True
                break

        if has_empty:
            rejected_videos.append(video_meta["video_name"])
        else:
            accepted_videos.append(video_meta["video_name"])

        if i%100==0:
            print(f"{i}/{len(data['annotations'])}: {len(accepted_videos)}/{len(rejected_videos)+len(accepted_videos)}")
        
    print(f"{len(accepted_videos)}/{len(rejected_videos)+len(accepted_videos)}")



    acc_rej_data = {"accepted": sorted(accepted_videos), "rejected": sorted(rejected_videos)}
    with open(os.path.join(root,'empty_videos.json'), 'w') as f:
        json.dump(acc_rej_data, f)






data_to_split = [
                ('MVPd/train/panoptic_train.json', 'MVPd/train')
                # ('MVPd/val/panoptic_val.json', 'MVPd/val'),
                # ('MVPd/test/panoptic_test.json', 'MVPd/test')
                ]
for data_src, label_root in data_to_split:
    data = json.load(open(data_src,'r'))
    reject_empty_videos(data, label_root)
