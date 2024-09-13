import csv

import numpy as np
from PIL import Image

import torch
import torchvision



rgb2hex = lambda r,g,b: '%02X%02X%02X' % (r,g,b)
hex2rgb = lambda hexs: list(int(hexs[i:i+2], 16) for i in (0, 2, 4))



def get_semantic_labels(file_path, scene_hex_color_to_category_map):
    semantic_label_image = Image.open(file_path).convert('RGB')
    semantic_label_image =  torchvision.transforms.functional.pil_to_tensor(semantic_label_image)
    if torch.cuda.is_available():
        semantic_label_image = semantic_label_image.cuda()

    semantic_rgb_colors = torch.unique(semantic_label_image.flatten(1,2).T, dim=0, sorted=True)
    semantic_hex_colors = [rgb2hex(*(rgb_color.cpu())) for rgb_color in semantic_rgb_colors]

    for rgb_color, hex_color in zip(semantic_rgb_colors, semantic_hex_colors):
        if hex_color not in scene_hex_color_to_category_map.keys() and hex_color!='000000':
            mask = torch.all(semantic_label_image == rgb_color.reshape(3,1,1), dim=0)
            semantic_label_image = torch.where(mask, 0, semantic_label_image)

    semantic_rgb_colors = torch.unique(semantic_label_image.flatten(1,2).T, dim=0, sorted=True)
    semantic_hex_colors = [rgb2hex(*(rgb_color.cpu())) for rgb_color in semantic_rgb_colors]
    semantic_object_names = [scene_hex_color_to_category_map[object_hex_color] for object_hex_color in semantic_hex_colors]
    semantic_mask = torch.all(semantic_label_image.unsqueeze(0) == semantic_rgb_colors.reshape(-1,3,1,1), dim=1).bool()
    
    return semantic_object_names, semantic_hex_colors, semantic_mask
    


def get_rgb_observation(file_path):
    rgba_image = np.array(Image.open(file_path))

    # Assign black to transparent pixels
    if rgba_image.shape[2]==4:
        rgba_image[rgba_image[...,-1]==0] = [0,0,0,0]
    rgb_image = Image.fromarray(rgba_image).convert("RGB")

    return rgb_image


def get_bbox_from_numpy_mask(mask):

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    x, y = cmin, rmin
    width, height = cmax-cmin, rmax-rmin

    return int(x), int(y), int(width), int(height)


#
# Matterport Specific Processes
#

def get_hex_color_to_category_map(SEMANTIC_SCENE_FILE_PATH):
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


def get_mpcat40_categories(MPCAT40_MAPPING_FILE_PATH):

    categories = []

    hex2rgb = lambda hexs: list(int(hexs[i:i+2], 16) for i in (0, 2, 4))

    with open(MPCAT40_MAPPING_FILE_PATH, 'r') as csvfile:

        cat_reader = csv.reader(csvfile, delimiter='\t')

        for cat_meta in cat_reader:
            # Skip first line
            if cat_meta[0]=='mpcat40index':
                continue

            # Skip void and unlabeled categories
            if cat_meta[0]=='0' or cat_meta[0]=='41':
                continue
            
            mpcat40_index = cat_meta[0]
            mpcat40_name = cat_meta[1]
            mpcat40_hex = cat_meta[2]

            categories.append({"id": int(mpcat40_index),
                                "name": str(mpcat40_name),
                                "isthing": 1,
                                "color": hex2rgb(mpcat40_hex.strip('#'))})
 
    return categories


def get_mpcat40_from_raw_category(raw_object_category, matterport_category_maps):
    raw_category_to_category_mapping, category_to_mpcat40_mapping = matterport_category_maps

    if raw_object_category in category_to_mpcat40_mapping:
        mpcat40_semantic_info = category_to_mpcat40_mapping[raw_object_category]
    
    elif raw_object_category in raw_category_to_category_mapping:
        mpcat40_semantic_info = category_to_mpcat40_mapping[raw_category_to_category_mapping[raw_object_category]]
    
    else:
        mpcat40_semantic_info = (0, 'void')

    return mpcat40_semantic_info



def get_raw_category_to_mpcat40_map(CATEGORY_TO_MPCAT40_MAPPING_FILE_PATH):

    # RAW_CATEGORY_TO_MPCAT40_MAPPING represents a many to one mapping to identify the keys to use for mapping labeled categories to matterport categories
    category_to_mpcat40_mapping = {}

    # RAW_CATEGORY_TO_CATEGORY_MAPPING represents a many to one mapping to identify the keys to use for mapping raw categories to labeled cattegories
    raw_category_to_category_mapping ={}
    
    with open(CATEGORY_TO_MPCAT40_MAPPING_FILE_PATH, 'r') as csvfile:

        map_reader = csv.reader(csvfile, delimiter='\t')

        for object_meta in map_reader:
            if object_meta[0]=='index':
                continue
            # print(object_meta)
            
            object_raw_category = object_meta[1]
            object_category = object_meta[2]
            # print(object_meta[16])
            object_mpcat40index = int(object_meta[16])
            object_mpcat40 = object_meta[17]

            if object_raw_category not in raw_category_to_category_mapping:
                raw_category_to_category_mapping[object_raw_category] = object_category
            else:
                assert raw_category_to_category_mapping[object_raw_category]==object_category

            if object_category in category_to_mpcat40_mapping:
                assert category_to_mpcat40_mapping[object_category] == (object_mpcat40index, object_mpcat40)
            else:
                category_to_mpcat40_mapping[object_category] = (object_mpcat40index, object_mpcat40)
 

    return raw_category_to_category_mapping, category_to_mpcat40_mapping

