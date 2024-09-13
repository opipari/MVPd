import os
import sys
import argparse
import configparser
import shutil
import csv

import math
import numpy as np
from PIL import Image
import torch
import torchvision
import itertools, functools, operator



if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='postprocess_dataset',
                    usage='python <path to postprocess_dataset.py> -- [options]',
                    description='Python script for rendering color images under active spot light illumination from the Matterport 3D semantic dataset assuming valid views have already been sampled',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args(argv)


    if args.verbose:
        print()
        print(args)
        print()

    

    rgb2hex = lambda r,g,b: '%02X%02X%02X' % (r,g,b)
    hex2rgb = lambda hex: tuple(int(h[i:i+2], 16) for i in (0, 2, 4))



        

    objects_in_sequence = {}
    num_sequences = 0
    num_frames = 0
    dist_traveled = 0

    scene_directories = sorted([path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))])
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)
        SCENE_DIR = scene_dir_path

        SEMANTIC_SCENE_FILE_PATH = os.path.join(scene_dir_path, scene_dir+'.semantic.txt')
        if not os.path.isfile(SEMANTIC_SCENE_FILE_PATH):
            continue

        SCENE_VIEWS_FILE = os.path.join(scene_dir_path, scene_dir+'.render_view_poses.csv')


        hex_color_to_category = {}
        scene_hex_colors = []
        with open(SEMANTIC_SCENE_FILE_PATH, "r") as sem_file:
            for line in sem_file.readlines():
                if line.startswith("HM3D Semantic Annotations"):
                    continue
                
                object_id, object_hex_color, object_name, unknown = line.split(',')
                object_name = object_name.strip('"')
                
                # if object_name not in scene_semantic_objects:
                #     scene_semantic_objects[object_name] = {}
                # if scene_dir not in scene_semantic_objects[object_name]:
                #     scene_semantic_objects[object_name][scene_dir] = []
                # scene_semantic_objects[object_name][scene_dir].append(object_hex_color)

                # assert object_hex_color not in hex_color_to_category.keys(), SEMANTIC_SCENE_FILE_PATH+"  "+str(object_hex_color)
                hex_color_to_category[object_hex_color] = object_name.strip('"')
                scene_hex_colors.append(object_hex_color)


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
                    prev_position = None
                    num_sequences += 1

                num_frames += 1

                if prev_position is not None:
                    x_prev, y_prev, z_prev = prev_position
                    dist_traveled += math.sqrt((x_pos-x_prev)**2 + (y_pos-y_prev)**2 + (z_pos-z_prev)**2)

                prev_position = (x_pos, y_pos, z_pos)

                sem_file = f"{'.'.join(info_ID)}.SEM.png"

                if os.path.exists(os.path.join(SCENE_DIR, sem_file)):
                    semantic_label_image = Image.open(os.path.join(SCENE_DIR, sem_file)).convert('RGB')
                    semantic_label_image =  torchvision.transforms.functional.pil_to_tensor(semantic_label_image)
                    if torch.cuda.is_available():
                        semantic_label_image = semantic_label_image.cuda()

                    semantic_label_image_rgb_colors = torch.unique(semantic_label_image.flatten(1,2).T, dim=0, sorted=True)
                    semantic_label_image_hex_colors = [rgb2hex(*(rgb_color.cpu())) for rgb_color in semantic_label_image_rgb_colors]

                    for rgb_color, hex_color in zip(semantic_label_image_rgb_colors, semantic_label_image_hex_colors):
                        if hex_color not in scene_hex_colors and hex_color!='000000':
                            mask = torch.all(semantic_label_image == rgb_color.reshape(3,1,1), dim=0)
                            semantic_label_image = torch.where(mask, 0, semantic_label_image)

                    semantic_label_image_rgb_colors = torch.unique(semantic_label_image.flatten(1,2).T, dim=0, sorted=True)
                    semantic_label_image_hex_colors = [rgb2hex(*(rgb_color.cpu())) for rgb_color in semantic_label_image_rgb_colors]
                    # semantic_label_mask = torch.all(semantic_label_image.unsqueeze(0) == semantic_label_image_rgb_colors.reshape(-1,3,1,1), dim=1).bool()
                    

                    if '000000' in semantic_label_image_hex_colors:
                        semantic_label_image_hex_colors.remove('000000')
                    if SEQ_NAME not in objects_in_sequence:
                        objects_in_sequence[SEQ_NAME] = set(semantic_label_image_hex_colors)
                    else:
                        objects_in_sequence[SEQ_NAME].update(set(semantic_label_image_hex_colors))

        print("Done with", scene_dir)
        print("Number of videos:", num_sequences)
        print("Number of frames:", num_frames)
        print("Frames per video:", num_frames/num_sequences)
        print("Distance Traveled:", dist_traveled)
        print("Distance per video:", dist_traveled / num_sequences)
        print("Objects Total:", sum([len(objects_in_sequence[seq]) for seq in objects_in_sequence]))
        print("Objects per Video:", sum([len(objects_in_sequence[seq]) for seq in objects_in_sequence])/len(objects_in_sequence.keys()))
        print()

    print("Number of videos:", num_sequences)
    print("Number of frames:", num_frames)
    print("Frames per video:", num_frames/num_sequences)
    print("Distance Traveled:", dist_traveled)
    print("Distance per video:", dist_traveled / num_sequences)
    print("Objects Total:", sum([len(objects_in_sequence[seq]) for seq in objects_in_sequence]))
    print("Objects per Video:", sum([len(objects_in_sequence[seq]) for seq in objects_in_sequence])/len(objects_in_sequence.keys()))
            

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE RENDERING ALL SCENES")
        print("***********************")
        print()