import os
import sys
import argparse
import configparser
import shutil
import csv
import json

import math
import numpy as np
from PIL import Image
import itertools, functools, operator

from panopticapi.utils import IdGenerator, save_json

import tqdm

import torch
import torchvision

from utils import rgb2hex, hex2rgb, get_semantic_labels, get_rgb_observation, get_bbox_from_numpy_mask
from utils import get_hex_color_to_category_map, get_mpcat40_categories, get_mpcat40_from_raw_category, get_raw_category_to_mpcat40_map



def format_scenes(scene_directories, 
            HM3DCat_path=os.path.join(os.getcwd(), './simulators/mpcat40.tsv'), 
            HM3DMap_path=os.path.join(os.getcwd(), './simulators/matterport_category_mappings.tsv')
    ):

    coco_categories = get_mpcat40_categories(HM3DCat_path)


    matterport_category_maps = get_raw_category_to_mpcat40_map(HM3DMap_path)

    for scene_dir in tqdm.tqdm(scene_directories):
        print(scene_dir)
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)

        scene_views_file = os.path.join(scene_dir_path, scene_dir+'.render_view_poses.csv')
        scene_hex_color_to_category_map = get_hex_color_to_category_map(os.path.join(scene_dir_path, scene_dir+'.semantic.txt'))

        OUT_DIR = os.path.join(args.output_dir, scene_dir)
        os.makedirs(OUT_DIR, exist_ok=True)
        
        coco_total_images = 0

        coco_videos = []
        coco_annotations = []
        coco_instances = []
        
        current_video_ID_str = None 
        video_images = []
        video_annotations = []

        with open(scene_views_file, 'r') as csvfile:

            pose_reader = csv.reader(csvfile, delimiter=',')

            for pose_idx, pose_meta in enumerate(pose_reader):
                info_ID = pose_meta[:3]
                view_ID = pose_meta[3]
                info_position = pose_meta[4:7]
                info_rotation = pose_meta[7:11]

                # Skip information line if it is first
                if info_ID[0]=='Scene-ID':
                    continue


                coco_total_images += 1
                video_ID_str = '.'.join(info_ID)
                if video_ID_str != current_video_ID_str:

                    # Save previous video data if on new video
                    assert len(video_images)==len(video_annotations)
                    assert len(coco_videos)==len(coco_annotations)
                    if current_video_ID_str is not None:
                        video_ID_int = len(coco_videos)
                        coco_videos.append({
                            "video_id": video_ID_int,
                            "images": video_images,
                            "video_name": current_video_ID_str,
                            })
                        coco_annotations.append({
                            "video_id": video_ID_int,
                            "annotations": video_annotations,
                            "video_name": current_video_ID_str,
                            })


                    # Update variables for new video data
                    current_video_ID_str = video_ID_str
                    # OUT_RGB_DIR = os.path.join(OUT_DIR, "imagesRGB", video_ID_str)
                    for rgb_dir in [fldr for fldr in os.listdir(scene_dir_path) if "imagesRGB" in fldr]:
                        os.makedirs(os.path.join(OUT_DIR, rgb_dir, video_ID_str), exist_ok=True)

                    OUT_SEM_DIR = os.path.join(OUT_DIR, "panomasksRGB", video_ID_str)
                    OUT_DEPTH_DIR = os.path.join(OUT_DIR, "imagesDEPTH", video_ID_str)

                    
                    os.makedirs(OUT_SEM_DIR, exist_ok=True)
                    os.makedirs(OUT_DEPTH_DIR, exist_ok=True)

                    coco_id_generator = IdGenerator({el['id']: el for el in coco_categories})
                    coco_video_instance_id_2_color = {}

                    video_images = []
                    video_annotations = []
                
                ###
                ### Depth Image
                ###
                depth_file = f"{view_ID}.png"
                # depth_file_in= f"{video_ID_str}.{view_ID}.DEPTH.png"
                # if os.path.exists(os.path.join(scene_dir_path, "imagesDEPTH", video_ID_str, depth_file_in)):
                #     os.rename(os.path.join(scene_dir_path, "imagesDEPTH", video_ID_str, depth_file_in), os.path.join(scene_dir_path, "imagesDEPTH", video_ID_str, depth_file))
                # else:
                #     assert os.path.exists(os.path.join(scene_dir_path, "imagesDEPTH", video_ID_str, depth_file))
                shutil.copyfile(os.path.join(scene_dir_path, "imagesDEPTH", video_ID_str, depth_file), os.path.join(OUT_DEPTH_DIR, depth_file))


                ###
                ### RGB Image
                ###
                for rgb_dir in [fldr for fldr in os.listdir(scene_dir_path) if "imagesRGB" in fldr]:
                    # rgb_file_in= f"{video_ID_str}.{view_ID}.RGB.0000000000.png"
                    # if os.path.exists(os.path.join(scene_dir_path, rgb_dir, video_ID_str, rgb_file_in)):
                    #     os.rename(os.path.join(scene_dir_path, rgb_dir, video_ID_str, rgb_file_in), os.path.join(scene_dir_path, rgb_dir, video_ID_str, f"{view_ID}.png"))
                    # else:
                    #     assert os.path.exists(os.path.join(scene_dir_path, rgb_dir, video_ID_str, f"{view_ID}.png"))
                    rgb_image = get_rgb_observation(os.path.join(scene_dir_path, rgb_dir, video_ID_str, f"{view_ID}.png"))
                    rgb_image.save(os.path.join(OUT_DIR, rgb_dir, video_ID_str, f"{view_ID}.jpg"))
                    rgb_width, rgb_height = rgb_image.size
                video_images.append({
                    "id":               int(coco_total_images),
                    "file_name":        str(f"{view_ID}.jpg"),
                    "width":            int(rgb_width),
                    "height":           int(rgb_height),
                    "scene_id":         str(scene_dir),
                    "camera_position":  [float(data_el) for data_el in info_position], # x_pos, y_pos, z_pos
                    "camera_rotation":  [float(data_el) for data_el in info_rotation], # quat_w, quat_x, quat_y, quat_z
                    "depth_file_name":  str(depth_file),
                    })

                ###
                ### Semantic Image
                ###
                sem_file = f"{view_ID}.png"
                # sem_file_in = f"{video_ID_str}.{view_ID}.SEM.png"
                # if os.path.exists(os.path.join(scene_dir_path, "panomasksRGB", video_ID_str, sem_file_in)):
                #     os.rename(os.path.join(scene_dir_path, "panomasksRGB", video_ID_str, sem_file_in), os.path.join(scene_dir_path, "panomasksRGB", video_ID_str, sem_file))
                # else:
                #     assert os.path.exists(os.path.join(scene_dir_path, "panomasksRGB", video_ID_str, sem_file))
                semantic_labels = get_semantic_labels(os.path.join(scene_dir_path, "panomasksRGB", video_ID_str, sem_file), scene_hex_color_to_category_map)


                pan_format = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)

                segments_info = []
                for instance_name, instance_hex_color, instance_mask in zip(*semantic_labels):
                    mpcat40_index, mpcat40_name = get_mpcat40_from_raw_category(instance_name, matterport_category_maps)

                    # Skip empty regions or unlabeled void
                    if instance_hex_color=='000000' or mpcat40_index<1 or mpcat40_index>40:
                        continue

                    if instance_hex_color not in coco_video_instance_id_2_color:
                        segment_ID, segment_color = coco_id_generator.get_id_and_color(mpcat40_index)
                        
                        instance_ID = len(coco_instances)+1
                        coco_instances.append({
                            "id":           int(instance_ID),
                            "category_id":  int(mpcat40_index),
                            "raw_category": str(instance_name),
                            "color":        list(hex2rgb(instance_hex_color)),
                            "scene_name":   str(scene_dir),
                            "video_name":   current_video_ID_str,
                            })

                        coco_video_instance_id_2_color[instance_hex_color] = (segment_ID, segment_color, instance_ID)
                    else:
                        segment_ID, segment_color, instance_ID = coco_video_instance_id_2_color[instance_hex_color]

                    pan_format[np.array(instance_mask.cpu())] = segment_color

                    segments_info.append({
                        "id":           int(segment_ID),
                        "category_id":  int(mpcat40_index),       
                        "area":         int(torch.sum(instance_mask).item()),
                        "bbox":         list(get_bbox_from_numpy_mask(np.array(instance_mask.cpu()))),
                        "iscrowd":      0,
                        "instance_id":  int(instance_ID)
                        })

                Image.fromarray(pan_format).save(os.path.join(OUT_SEM_DIR, str(view_ID+".png")))
                
                video_annotations.append({
                        "image_id"      : int(coco_total_images),
                        "file_name"     : str(view_ID+".png"),
                        "segments_info" : list(segments_info)
                        })


        # Save previous video data if on new video
        assert len(video_images)==len(video_annotations)
        assert len(coco_videos)==len(coco_annotations)
        if current_video_ID_str is not None:
            video_ID_int = len(coco_videos)
            coco_videos.append({
                "video_id": video_ID_int,
                "images": video_images,
                "video_name": current_video_ID_str,
                })
            coco_annotations.append({
                "video_id": video_ID_int,
                "annotations": video_annotations,
                "video_name": current_video_ID_str,
                })

        coco_annotations = {'videos': coco_videos,
                        'annotations': coco_annotations,
                        'instances': coco_instances,
                        'categories': coco_categories,
                        }

        with open(os.path.join(OUT_DIR, f"panoptic.json"), "w") as outfile:
            json.dump(coco_annotations, outfile)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='format_HM3DSeg_2_VIPOSeg',
                    usage='python <path to format_HM3DSeg_2_VIPOSeg.py> -- [options]',
                    description='Python script for converting format of rendered data from Matterport scans into VIPOSeg format needed for PAOT Benchmark',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-data', '--dataset-dir', help='path to directory of rendered HM3D images', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args()


    if args.verbose:
        print()
        print(args)
        print()

    

    

    # 00832-qyAac8rV8Zk

    scene_directories = sorted([path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))])
    scene_directories = [
                         '00087-YY8rqV6L6rf',
                         # '00598-mt9H8KcxRKD',
                         # '00624-ooq3SnvC79d',
                         # '00638-iePHCSf119p',
                         # '00732-Z2DQddYp1fn'
                        ]
    # scene_directories = [scene for scene in scene_directories if scene not in already_scenes]
    print(len(scene_directories))
    
    from multiprocessing import Pool

    format_scenes(scene_directories)
    # with Pool(2) as p:
    #     p.map(format_scenes, [scene_directories[0::2], 
    #                             scene_directories[1::2]])

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE FORMATTING ALL SCENES")
        print("***********************")
        print()