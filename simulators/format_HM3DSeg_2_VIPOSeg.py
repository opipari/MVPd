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

import torch
import torchvision





def hex_color_to_category_map(SEMANTIC_SCENE_FILE_PATH):
    hex_color_to_category_map = {'000000': 'void'}
    with open(SEMANTIC_SCENE_FILE_PATH, "r") as sem_file:
        for line in sem_file.readlines():
            if line.startswith("HM3D Semantic Annotations"):
                continue
            
            object_id, object_hex_color, object_name, unknown = line.split(',')
            object_name = object_name.strip('"')

            # assert object_hex_color not in hex_color_to_category_map.keys()
            hex_color_to_category_map[object_hex_color] = object_name.strip('"')

    return hex_color_to_category_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='format_HM3DSeg_2_VIPOSeg',
                    usage='python <path to format_HM3DSeg_2_VIPOSeg.py> -- [options]',
                    description='Python script for converting format of rendered data from Matterport scans into VIPOSeg format needed for PAOT Benchmark',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-data', '--dataset-dir', help='path to directory of rendered HM3D images', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-mode', '--training-mode', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args()


    if args.verbose:
        print()
        print(args)
        print()

    

    rgb2hex = lambda r,g,b: '%02X%02X%02X' % (r,g,b)
    hex2rgb = lambda hexs: tuple(int(hexs[i:i+2], 16) for i in (0, 2, 4))

    OUT_DIR = args.output_dir

    
    RAW_CATEGORY_TO_CATEGORY_MAPPING, CATEGORY_TO_MPCAT40_MAPPING = get_raw_category_to_mpcat40_map()
    if 'void' in CATEGORY_TO_MPCAT40_MAPPING:
        assert CATEGORY_TO_MPCAT40_MAPPING['void']==(0, 'void')
    else:
        CATEGORY_TO_MPCAT40_MAPPING['void'] = (0, 'void')

    obj_class_map = {}
    meta = {'videos': {}}
    scene_directories = sorted([path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))])
    scene_directories = [scene_directories[0]]
    print(scene_directories)
    tot_scenes = 0
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)
        SCENE_DIR = scene_dir_path

        scene_view_poses_path = os.path.join(scene_dir_path, scene_dir+'.render_view_poses.csv')
        scene_has_sampled_views = os.path.isfile(scene_view_poses_path)

        SEMANTIC_SCENE_FILE_PATH = os.path.join(scene_dir_path, scene_dir+'.semantic.txt')
        if not os.path.isfile(SEMANTIC_SCENE_FILE_PATH):
            continue

        print(scene_dir)
        SCENE_VIEWS_FILE = os.path.join(scene_dir_path, scene_dir+'.render_view_poses.csv')

        tot_scenes+=1


        
        scene_hex_color_to_category_map = get_hex_color_to_category_map(SEMANTIC_SCENE_FILE_PATH)

        
        extracting_sequence = None 

        with open(SCENE_VIEWS_FILE, 'r') as csvfile:

            pose_reader = csv.reader(csvfile, delimiter=',')

            for pose_idx, pose_meta in enumerate(pose_reader):
                info_ID = pose_meta[:4]
                info_position = pose_meta[4:7]
                info_rotation = pose_meta[7:11]

                # Skip information line if it is first
                if info_ID[0]=='Scene-ID':
                    continue

                scene_id, trajectory_id, sensor_height_id, view_id = info_ID
                x_pos, y_pos, z_pos = info_position
                quat_w, quat_x, quat_y, quat_z = info_rotation

                # Parse pose infomration out of string type
                x_pos, y_pos, z_pos = float(x_pos), float(y_pos), float(z_pos)
                quat_w, quat_x, quat_y, quat_z = float(quat_w), float(quat_x), float(quat_y), float(quat_z)


                if (scene_id, trajectory_id, sensor_height_id) != extracting_sequence:
                    extracting_sequence = (scene_id, trajectory_id, sensor_height_id)
                    SEQ_NAME = '.'.join(extracting_sequence)
                    
                    OUT_RGB_DIR = os.path.join(OUT_DIR, "JPEGImages", SEQ_NAME)
                    OUT_SEM_DIR = os.path.join(OUT_DIR, "Annotations", SEQ_NAME)
                    OUT_SEM_GT_DIR = os.path.join(OUT_DIR, "Annotations_gt", SEQ_NAME)

                    os.makedirs(OUT_RGB_DIR, exist_ok=True)
                    os.makedirs(OUT_SEM_DIR, exist_ok=True)
                    os.makedirs(OUT_SEM_GT_DIR, exist_ok=True)

                    object_ids_in_seq = set()

                    sequence_object_hex_color_to_id = {}
                    obj_class_map[SEQ_NAME] = {}
                    meta['videos'][SEQ_NAME] = {'objects': {}}



                sem_file = f"{'.'.join(info_ID)}.SEM.png"
                semantic_label_image = Image.open(os.path.join(SCENE_DIR, sem_file)).convert('RGB')
                semantic_label_image =  torchvision.transforms.functional.pil_to_tensor(semantic_label_image)
                if torch.cuda.is_available():
                    semantic_label_image = semantic_label_image.cuda()

                semantic_label_image_rgb_colors = torch.unique(semantic_label_image.flatten(1,2).T, dim=0, sorted=True)
                semantic_label_image_hex_colors = [rgb2hex(*(rgb_color.cpu())) for rgb_color in semantic_label_image_rgb_colors]

                for rgb_color, hex_color in zip(semantic_label_image_rgb_colors, semantic_label_image_hex_colors):
                    if hex_color not in scene_hex_color_to_category_map.keys() and hex_color!='000000':
                        mask = torch.all(semantic_label_image == rgb_color.reshape(3,1,1), dim=0)
                        semantic_label_image = torch.where(mask, 0, semantic_label_image)

                semantic_label_image_rgb_colors = torch.unique(semantic_label_image.flatten(1,2).T, dim=0, sorted=True)
                semantic_label_image_hex_colors = [rgb2hex(*(rgb_color.cpu())) for rgb_color in semantic_label_image_rgb_colors]
                semantic_label_image_object_names = [scene_hex_color_to_category_map[object_hex_color] for object_hex_color in semantic_label_image_hex_colors]
                semantic_label_mask = torch.all(semantic_label_image.unsqueeze(0) == semantic_label_image_rgb_colors.reshape(-1,3,1,1), dim=1).bool()
                

                
                category_ids = []
                instance_ids = []
                is_new_instance = []
                for object_hex_color, object_name in zip(semantic_label_image_hex_colors, semantic_label_image_object_names):
                    if object_name in CATEGORY_TO_MPCAT40_MAPPING:
                        class_id = CATEGORY_TO_MPCAT40_MAPPING[object_name][0]
                        category_ids.append(class_id)
                        
                        is_new_inst = 0
                        if class_id!=0:
                            if object_hex_color not in sequence_object_hex_color_to_id:
                                sequence_object_hex_color_to_id[object_hex_color] = len(sequence_object_hex_color_to_id.keys())+1
                                is_new_inst = 1
                            instance_id = sequence_object_hex_color_to_id[object_hex_color]

                            obj_class_map[SEQ_NAME][str(instance_id)] = str(class_id)
                        else:
                            instance_id = 0
                        instance_ids.append(instance_id)
                        is_new_instance.append(is_new_inst)

                    elif object_name in RAW_CATEGORY_TO_CATEGORY_MAPPING:
                        class_id = CATEGORY_TO_MPCAT40_MAPPING[RAW_CATEGORY_TO_CATEGORY_MAPPING[object_name]][0]
                        category_ids.append(class_id)
                        
                        if class_id!=0:
                            if object_hex_color not in sequence_object_hex_color_to_id:
                                sequence_object_hex_color_to_id[object_hex_color] = len(sequence_object_hex_color_to_id.keys())+1
                                is_new_inst = 1
                            instance_id = sequence_object_hex_color_to_id[object_hex_color]

                            obj_class_map[SEQ_NAME][str(instance_id)] = str(class_id)
                        else:
                            instance_id = 0
                        instance_ids.append(instance_id)
                        is_new_instance.append(is_new_inst)
                    else:
                        class_id = 0
                        category_ids.append(class_id)

                        instance_id = 0
                        instance_ids.append(instance_id)

                        is_new_inst = 0
                        is_new_instance.append(is_new_inst)

                # print(max(instance_ids), len(instance_ids), SEQ_NAME)
                if max(instance_ids)>255:
                    continue


                rgb_file = f"{'.'.join(info_ID)}.RGB.{0:010}.png"
                rgb_image = Image.open(os.path.join(SCENE_DIR, rgb_file)).convert("RGB")
                rgb_image.save(os.path.join(OUT_RGB_DIR, view_id+".jpg"))


                for inst_id in instance_ids:
                    if inst_id==0:
                        continue

                    if str(inst_id) not in meta['videos'][SEQ_NAME]['objects']:
                        meta['videos'][SEQ_NAME]['objects'][str(inst_id)] = {'frames': [], 'color': list(sequence_object_hex_color_to_id.keys())[list(sequence_object_hex_color_to_id.values()).index(inst_id)]}
                    meta['videos'][SEQ_NAME]['objects'][str(inst_id)]['frames'].append(view_id)

                
                

                _palette = [[0,0,0]]+[hex2rgb(hex_color) for hex_color in sorted(sequence_object_hex_color_to_id, key=lambda k: sequence_object_hex_color_to_id[k])]
                _palette = list(itertools.chain.from_iterable(_palette))

                if args.training_mode=='train':
                    panomask = torch.sum(torch.tensor(instance_ids).reshape(-1,1,1).to(semantic_label_mask.device) * semantic_label_mask, dim=0).cpu()
                    panomask = Image.fromarray(np.array(panomask).astype(np.uint8))
                    panomask = panomask.convert('P')
                    panomask.putpalette(_palette)
                    panomask.save(os.path.join(OUT_SEM_DIR, view_id+".png"))
                    
                else:
                    if any(is_new_instance):
                        panomask = torch.sum(torch.tensor(is_new_instance).reshape(-1,1,1).to(semantic_label_mask.device) * torch.tensor(instance_ids).reshape(-1,1,1).to(semantic_label_mask.device) * semantic_label_mask, dim=0).cpu()
                        panomask = Image.fromarray(np.array(panomask).astype(np.uint8))
                        panomask = panomask.convert('P')
                        panomask.putpalette(_palette)
                        panomask.save(os.path.join(OUT_SEM_DIR, view_id+".png"))

                    panomask = torch.sum(torch.tensor(instance_ids).reshape(-1,1,1).to(semantic_label_mask.device) * semantic_label_mask, dim=0).cpu()
                    panomask = Image.fromarray(np.array(panomask).astype(np.uint8))
                    panomask = panomask.convert('P')
                    panomask.putpalette(_palette)
                    panomask.save(os.path.join(OUT_SEM_GT_DIR, view_id+".png"))

    with open(os.path.join(OUT_DIR, "obj_class.json"), "w") as outfile:
        json.dump(obj_class_map, outfile)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as outfile:
        json.dump(meta, outfile)



                

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE RENDERING ALL SCENES")
        print("***********************")
        print()