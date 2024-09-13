
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

    dataset_dict = {
        "categories": [{'id': 0, 'name': 'object'}],
        "videos": [],
        "annotations": []
    }

    inst_id = 0
    for video_i, video in enumerate(tqdm(dataset)):
        video_record = {}
        video_record['id'] = video_i
        video_record['file_names'] = []


        sample_0 = video[0]
        video_record['height'] = sample_0['observation']['image'].shape[1]
        video_record['width'] = sample_0['observation']['image'].shape[2]
        video_record['length'] = len(video)

        instance_ids = list(sample_0['meta']['class_dict'].keys())
        instance_ids = [iid for iid in instance_ids if iid!=0]

        inst_annotations = []
        for inst in instance_ids:
            inst_annotations.append({'video_id': video_i,
                'iscrowd': '0',
                'height': video_record['height'],
                'width': video_record['width'],
                'length': video_record['length'],
                'segmentations': [],
                'category_id': 0,
                'id': inst_id,
                'areas': []
                })
            inst_id += 1

        for sample_i, sample in enumerate(video):
            
            video_name = sample['meta']['video_name']
            image_name = sample['meta']['window_names'][0]
            mask_anno = sample['label']['mask'][0]
            
            video_record['file_names'].append(os.path.join(args.root_path, args.split, 'imagesRGB', video_name, image_name))
            
            for inst_i, seg_id in enumerate(instance_ids):
                mask_bin = mask_anno == seg_id
                
                rle = COCOmask.encode(np.asfortranarray(mask_bin.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('utf8')
               
                inst_annotations[inst_i]['areas'].append(int(mask_bin.sum()))
                inst_annotations[inst_i]['segmentations'].append(rle) # cfg.INPUT.MASK_FORMAT must be set to bitmask
                

        dataset_dict["videos"].append(video_record)    
        dataset_dict["annotations"] = dataset_dict["annotations"] + inst_annotations
        
        

    with open(f"MVPd_{args.split}_coco.json", "w") as outf:
        json.dump(dataset_dict, outf)



