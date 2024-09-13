import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torchvision.ops import masks_to_boxes
from pycocotools import mask as COCOmask
from detectron2.structures import BoxMode
from MVPd.utils.MVPdataset import MVPDataset, MVPdCategories


def format_cursor_data(data):
    return "[" + str(data) + "]"



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--use_stuff', action='store_true', default=False)
    parser.add_argument('--use_categories', action='store_true', default=False)
    parser.add_argument('--empty_videos_split', action='store_false', default=True)
    parser.add_argument('--zero_shot_split', action='store_false', default=True)
    args = parser.parse_args()


    dataset = MVPDataset(root=args.root_path,
                    split=args.split,
                    window_size = -1,
                    use_stuff = args.use_stuff,
                    empty_videos_split = args.empty_videos_split,
                    zero_shot_split = args.zero_shot_split
                    )


    im_id = 0
    dataset_dicts = []
    for video in tqdm(dataset):
        for sample in video:
            record = {}

            video_name = sample['meta']['video_name']
            image_name = sample['meta']['window_names'][0]
            mask_anno = sample['label']['mask'][0]

            record['file_name'] = os.path.join(args.root_path, args.split, 'imagesRGB', video_name, image_name)
            record['height'] = sample['observation']['image'].shape[1]
            record['width'] = sample['observation']['image'].shape[2]
            record['image_id'] = im_id

            annos = []
            for seg_id in np.unique(mask_anno):
                if seg_id==0:
                    continue
                anno_info = {}

                mask_bin = mask_anno == seg_id
                
                rle = COCOmask.encode(np.asfortranarray(mask_bin.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('utf8')
                anno_info['segmentation'] = rle # cfg.INPUT.MASK_FORMAT must be set to bitmask
                anno_info['bbox'] = [xy.item() for xy in masks_to_boxes(torch.tensor(mask_bin).unsqueeze(0))[0].int()]
                anno_info['bbox_mode'] = BoxMode.XYXY_ABS
                anno_info['category_id'] = sample['meta']['class_dict'][seg_id] if args.use_categories else 0
                annos.append(anno_info)

            record["annotations"] = annos
            dataset_dicts.append(record)    

            im_id += 1

    with open(f"MVPd_{args.split}_detectron2.json", "w") as outf:
        json.dump(dataset_dicts, outf)