import os
import json

from typing import List, Optional, Tuple, Callable, Union

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

import pytorch3d.transforms as tforms

from panopticapi.utils import rgb2id, IdGenerator
from .MVPdHelpers import get_RT_inverse, get_pytorch3d_matrix, filter_idmask_area


rgb2hex = lambda r,g,b: '%02X%02X%02X' % (r,g,b)


__all__ = [
    'video_collate',
    'MVPVideo',
    'MVPDataset',
    'MVPdCategories'
]

MVPdCategories = {
    0:   'void',
    1:   'wall',
    2:   'floor',
    3:   'chair',
    4:   'door',
    5:   'table',
    6:   'picture',
    7:   'cabinet',
    8:   'cushion',
    9:   'window',
    10:  'sofa',
    11:  'bed',
    12:  'curtain',
    13:  'chest_of_drawers',
    14:  'plant',
    15:  'sink',
    16:  'stairs',
    17:  'ceiling',
    18:  'toilet',
    19:  'stool',
    20:  'towel',
    21:  'mirror',
    22:  'tv_monitor',
    23:  'shower',
    24:  'column',
    25:  'bathtub',
    26:  'counter',
    27:  'fireplace',
    28:  'lighting',
    29:  'beam',
    30:  'railing',
    31:  'shelving',
    32:  'blinds',
    33:  'gym_equipment',
    34:  'seating',
    35:  'board_panel',
    36:  'furniture',
    37:  'appliances',
    38:  'clothes',
    39:  'objects',
    40:  'misc'
}


def video_collate(
    batch: dict
) -> dict:
    elem = batch[0]
    batched_sample = {}
    for k in elem.keys():
        batched_k = {}
        for km in elem[k].keys():
            if isinstance(elem[k][km], np.ndarray):
                batched_k[km] = torch.tensor(np.stack([b[k][km] for b in batch], axis=1))
            elif isinstance(elem[k][km], list):
                batched_k[km] = list(zip(*[b[k][km] for b in batch]))
            elif isinstance(elem[k][km], dict):
                batched_k[km] = [b[k][km] for b in batch]
            else:
                batched_k[km] = default_collate([b[k][km] for b in batch])
        
        batched_sample[k] = batched_k

    return batched_sample


class MVPVideo(Dataset):
    def __init__(
        self, 
        image_root: str,
        depth_root: str,
        label_root: str,
        video_meta: dict,
        anno_meta: dict,
        categories: dict,
        window_size: int = 1,
        transform: Optional[Callable[[dict], dict]] = None,
        rgb: bool = True,
        use_stuff: bool = True,
        filter_pcnt: float = 0.001,
        zero_shot_objects: dict = {},
    ) -> None:

        self.image_root = image_root
        self.depth_root = depth_root
        self.label_root = label_root
        self.video_meta = video_meta
        self.anno_meta = anno_meta
        self.categories = categories
        self.stuff_category_ids = [el['id'] for el in self.categories if not el['isthing']]
        self.window_size = window_size
        self.transform = transform
        self.rgb = rgb
        self.use_stuff = use_stuff
        self.filter_pcnt = filter_pcnt
        self.zero_shot_objects = zero_shot_objects

        self.panoptic_idgenerator = IdGenerator({el['id']: el for el in self.categories})

        # Note that MVPCameras were defined in blender, with -Z forward, +X right, and +Y up
        # Meanwhile, PyTorch3D defines its camera view coordinate system with +Z forward, -X right, and +Y up
        # Hence, to convert between these coordinate systems, we use a 180deg rotation about Y, given below:
        self.MVPCamera2PyTorch3dViewCoSys = torch.tensor([[[-1.0,  0.0,  0.0,  0.0],
                                                             [ 0.0,  1.0,  0.0,  0.0],
                                                             [ 0.0,  0.0, -1.0,  0.0],
                                                             [ 0.0,  0.0,  0.0,  1.0]]],
                                                             dtype=torch.float32)
        self.PyTorch3dViewCoSys2MVPCamera = self.MVPCamera2PyTorch3dViewCoSys


    def __len__(self) -> int:
        return len(self.video_meta["images"]) - self.window_size + 1


    def get_image_depth_label(
        self,
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_meta = self.video_meta["images"][idx]
        anno_meta = self.anno_meta["annotations"][idx]

        image = cv2.imread(
            os.path.join(self.image_root, self.video_meta["video_name"], image_meta["file_name"]))
        image = np.array(image, dtype=np.float32)

        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        depth_file = os.path.join(self.depth_root, self.video_meta["video_name"], '.'.join(image_meta["file_name"].split('.')[:-1])+'.png')
        if os.path.isfile(depth_file):
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            depth = np.array(depth, dtype=np.float32)/1000.0
        else:
            depth = []

        label = Image.open(
            os.path.join(self.label_root, self.video_meta["video_name"], anno_meta["file_name"]))
        label = np.array(label, dtype=np.uint8)
        label = rgb2id(label)

        return image, depth, label

    def get_pose_matrices(
        self,
        rotation_sequence: List,
        position_sequence: List
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Helper to convert quaternion and translation to homogeneous pose transformation matrices
        '''
        position_sequence_tensor = torch.tensor(position_sequence)
        rotation_sequence_tensor = torch.tensor(rotation_sequence)

        # Calculate rotation and translation from MVPCamera to World
        R_C2W = tforms.quaternion_to_matrix(rotation_sequence_tensor)
        T_C2W = position_sequence_tensor.reshape(-1,3,1)
        # Convert rotation and translation into pytorch3d matrix form (transposed homogeneous matrix)
        C2W = get_pytorch3d_matrix(R_C2W, T_C2W)
        # Convert transformation to be from PyTorch3d Camera View Coordinate System to World frame: https://pytorch3d.org/docs/cameras
        # NOTE: the C2W matrix is in transposed form; hence, the following matrix multiplication is in reversed order to compensate. 
        # I.E. if working out on paper, you'll expect: matmul(MVPCamera->World, PyTorch3dCamViewCoSys->MVPCamera),
        # but instead we do matmul(PyTorch3dCamViewCoSys->MVPCamera, MVPCamera->World.T)
        # This works since PyTorch3dCamViewCoSys->MVPCamera is its own inverse (transpose)
        V2W = torch.matmul(self.PyTorch3dViewCoSys2MVPCamera, C2W) # Note the order of mult. is flipped due to transposed M
        
        # Calculate rotation and translation from World to MVPCamera
        R_W2C, T_W2C = get_RT_inverse(R_C2W, T_C2W)
        # Convert rotation and translation into pytorch3d matrix form (transposed homogeneous matrix)
        W2C = get_pytorch3d_matrix(R_W2C, T_W2C)
        # Convert transformation to be from World to pytorch3d Camera View Coordinate System: https://pytorch3d.org/docs/cameras
        # NOTE: the W2C matrix is in transposed form; hence, the following matrix multiplication is in reversed order to compensate.
        # I.E. if working out on paper, you'll expect: matmul(MVPCamera->PyTorch3dCamViewCoSys, World->MVPCamera),
        # but instead we do matmul(World->World->MVPCamera.T, MVPCamera->PyTorch3dCamViewCoSys)
        # This works since MVPCamera->CamViewCoSys is its own inverse (transpose)
        W2V = torch.matmul(W2C, self.MVPCamera2PyTorch3dViewCoSys)
        
        return V2W, W2V 
    
    def merge_instances_of_stuff(
        self,
        instance_id_to_category_id: dict
    ) -> Tuple[dict, dict]:
        to_merge_instance_id_to_stuff_id = {}

        # Iterate over classes representing 'stuff', e.g. walls, ceilings, floors
        for stuff_cat_id in self.stuff_category_ids:
            # Identify every instance id belonging to a stuff category
            to_merge_instance_ids = [inst_id for inst_id, cat_id in instance_id_to_category_id.items() if cat_id==stuff_cat_id]
            to_merge_instance_ids = sorted(to_merge_instance_ids)

            # If any instances of a stuff category exist, assign them to a shared, canonical instance id
            # Effectively, this maps any instances labeled within a stuff class by HM3D annotators to a single stuff category
            # By default, HM3D annotated walls, floors and ceilings at the instance level. This operation maps those instances to stuff classes respectively.
            if len(to_merge_instance_ids)>0:
                assignment_id = self.panoptic_idgenerator.get_id(stuff_cat_id)
                for to_merge_id in to_merge_instance_ids:
                    to_merge_instance_id_to_stuff_id[to_merge_id] = assignment_id
                    del instance_id_to_category_id[to_merge_id]
                instance_id_to_category_id[assignment_id] = stuff_cat_id

        return instance_id_to_category_id, to_merge_instance_id_to_stuff_id

    def merge_stuff_labels(
        self,
        label: np.ndarray,
        to_merge_instance_id_to_stuff_id: dict
    ) -> Tuple[np.ndarray, dict]:
        
        for to_merge_id, assignment_id in to_merge_instance_id_to_stuff_id.items():
            label[label==to_merge_id] = assignment_id

        return label

    def __getitem__(
        self,
        idx: int
    ) -> dict:

        start_idx = idx
        end_idx = start_idx + self.window_size

        image_sequence = []
        depth_sequence = []
        label_sequence = []

        id_sequence = []

        position_sequence = []
        rotation_sequence = []

        sequence_class_dict = {int(key):int(value['category_id']) for key,value in self.anno_meta['instance_id_map'].items()}
        if self.use_stuff:
            sequence_class_dict, to_merge_instance_id_to_stuff_id = self.merge_instances_of_stuff(sequence_class_dict)
        
        sequence_zeroshot_dict = {}
        for key,value in self.anno_meta['instance_id_map'].items():
            if value['scene_name'] in self.zero_shot_objects:
                sequence_zeroshot_dict[int(key)] = rgb2hex(*value['color']) in self.zero_shot_objects[value['scene_name']]
            else:
                sequence_zeroshot_dict[int(key)] = False

        for sub_idx in range(start_idx, end_idx):
            image, depth, label = self.get_image_depth_label(sub_idx)

            if self.use_stuff:
                label = self.merge_stuff_labels(label, to_merge_instance_id_to_stuff_id)
            label = filter_idmask_area(label, filter_pcnt=self.filter_pcnt)
            
            image_sequence.append(image)
            depth_sequence.append(depth)
            label_sequence.append(label)

            id_sequence.append(list(np.sort(np.unique(label))))

            position_sequence.append(self.video_meta["images"][sub_idx]["camera_position"])
            rotation_sequence.append(self.video_meta["images"][sub_idx]["camera_rotation"])
        
            

        sample = {}

        sample['observation'] = {
            'image': np.array(image_sequence),
            'depth': np.array(depth_sequence)
        }


        fx = 465.6029
        fy = 465.6029
        cx = 320.00
        cy = 240.00
        V2W, W2V = self.get_pose_matrices(rotation_sequence, position_sequence)
        
        sample['camera'] = {
            'K': np.tile(np.array([[[fx,  0.0, cx,  0.0],
                                    [0.0, fy,  cy,  0.0],
                                    [0.0, 0.0, 0.0, 1.0], 
                                    [0.0, 0.0, 1.0, 0.0]]],
                                    dtype=np.float32),
                              (self.window_size,1,1)),
            'W2V_pose': np.array(W2V),
            'V2W_pose': np.array(V2W)
        }

        sample['label'] = {
            'mask': np.array(label_sequence),
            'id': list(id_sequence)
        }

        sample['meta'] = {
            'video_name': self.video_meta['video_name'],
            'window_idxs': np.array([sub_idx for sub_idx in range(start_idx, end_idx)]),
            'window_names': [self.video_meta['images'][sub_idx]["file_name"] for sub_idx in range(start_idx, end_idx)],
            'class_dict': sequence_class_dict,
            'zero_shot_dict': sequence_zeroshot_dict,
            'image_size': np.array([img.shape[:2] for img in image_sequence])
        }
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    


class MVPDataset(Dataset):
    def __init__(
        self,
        root: str = './datasets/MVPd',
        split: str = 'train',
        window_size: int = 1,
        image_root: str = 'imagesRGB',
        transform: Optional[Callable[[dict], dict]] = None,
        rgb: bool = True,
        use_stuff: bool = True,
        filter_pcnt: float = 0.001,
        empty_videos_split = True,
        zero_shot_split = True,
    ) -> None:

        self.root = root
        self.split = split
        self.window_size = window_size
        self.transform = transform
        self.rgb = rgb
        self.use_stuff = use_stuff
        self.filter_pcnt = filter_pcnt
        self.empty_videos_split = empty_videos_split
        self.zero_shot_split = zero_shot_split
        self.zero_shot_objects = {}

        self.image_root = os.path.join(root, f'{self.split}/{image_root}')
        self.depth_root = os.path.join(root, f'{self.split}/imagesDEPTH')
        self.label_root = os.path.join(root, f'{self.split}/panomasksRGB')

        annotation_file = os.path.join(root, f'{self.split}/panoptic_{self.split}.json')
        self.dataset = json.load(open(annotation_file, 'r'))
        if self.empty_videos_split:
            empty_split = json.load(open(os.path.join(root, f'{self.split}/empty_videos.json'), 'r'))
            self.dataset['videos'][:] = [vid for vid in self.dataset['videos'] if vid['video_name'] in empty_split['accepted']]
            self.dataset['annotations'][:] = [ann for ann in self.dataset['annotations'] if ann['video_name'] in empty_split['accepted']]
        
        if self.zero_shot_split and self.split=='train':
            zero_shot_split = json.load(open(os.path.join(root, f'{self.split}/zero_shot.json'), 'r'))
            self.dataset['videos'][:] = [vid for vid in self.dataset['videos'] if vid['video_name'] in zero_shot_split['accepted']]
            self.dataset['annotations'][:] = [ann for ann in self.dataset['annotations'] if ann['video_name'] in zero_shot_split['accepted']]
        elif self.zero_shot_split and (self.split=='val' or self.split=='test'):
            for el in json.load(open(os.path.join(root, f'{self.split}/zero_shot.json'), 'r'))['zero-shot']:
                el_scene_name = str(el['scene_name'])
                el_color = str(rgb2hex(*el['color']))
                if el_scene_name not in self.zero_shot_objects:
                    self.zero_shot_objects[el_scene_name] = []
                self.zero_shot_objects[el_scene_name].append(el_color)
        assert len(self.dataset['videos'])==len(self.dataset['annotations'])
        for i in range(len(self.dataset['videos'])):
            assert self.dataset['videos'][i]['video_name']==self.dataset['annotations'][i]['video_name']

        windows_per_video = [len(v['images'])-self.window_size+1 for v in self.dataset['videos']]
        self.cumulative_windows_per_video = np.cumsum(windows_per_video)
        print(f"{len(self.dataset['videos'])} videos in MVPd:{self.root}/{self.split}")

    def __len__(self) -> int:
        if self.window_size>0:
            return self.cumulative_windows_per_video[-1]
        else:
            return len(self.dataset["videos"])

    def __getitem__(
        self,
        idx: int
    ) -> Union[dict, MVPVideo]:
        if self.window_size>0:
            video_idx = np.argmax(idx<self.cumulative_windows_per_video)
            window_idx = idx if video_idx==0 else idx-self.cumulative_windows_per_video[video_idx-1]
            return MVPVideo(self.image_root, 
                            self.depth_root, 
                            self.label_root,
                            self.dataset['videos'][video_idx], 
                            self.dataset['annotations'][video_idx],
                            self.dataset['categories'],
                            window_size = self.window_size,
                            transform = self.transform,
                            rgb = self.rgb,
                            use_stuff = self.use_stuff,
                            filter_pcnt = self.filter_pcnt,
                            zero_shot_objects=self.zero_shot_objects
                            )[window_idx]

        else:
            return MVPVideo(self.image_root, 
                            self.depth_root, 
                            self.label_root,
                            self.dataset['videos'][idx], 
                            self.dataset['annotations'][idx],
                            self.dataset['categories'],
                            window_size = 1,
                            transform = self.transform,
                            rgb = self.rgb,
                            use_stuff = self.use_stuff,
                            filter_pcnt = self.filter_pcnt,
                            zero_shot_objects=self.zero_shot_objects
                            )