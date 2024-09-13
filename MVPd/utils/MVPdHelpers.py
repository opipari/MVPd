import os
from typing import Tuple, Type

import cv2
import numpy as np

import matplotlib.pyplot as plt

import torch
from pytorch3d.renderer import CamerasBase, PerspectiveCameras


__all__ = [
    'get_cameras',
    'get_xy_depth',
    'get_RT_inverse',
    'get_pytorch3d_matrix',
    'label_to_one_hot',
    'colorize_depth',
    'make_video'
]


def get_cameras(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_sizes: torch.Tensor
) -> Type[CamerasBase]:
    """
    Assumes input extrinsics follow pytorch3d matrix format
    M = [
        [Rxx, Ryx, Rzx, 0],
        [Rxy, Ryy, Rzy, 0],
        [Rxz, Ryz, Rzz, 0],
        [Tx,  Ty,  Tz,  1],
    ]
    """
    assert intrinsics.shape[0]==extrinsics.shape[0]==image_sizes.shape[0]
    assert intrinsics.shape[1:]==(4,4)
    assert extrinsics.shape[1:]==(4,4)
    assert image_sizes.shape[1:]==(2,)
    
    return PerspectiveCameras(in_ndc=False, K=intrinsics, R=extrinsics[:,:3,:3], T=extrinsics[:,3,:3], image_size=image_sizes)


def get_xy_depth(
    depth_map: torch.Tensor, # B x 1 x H x W
    from_ndc: bool = False
) -> torch.Tensor:
    # assert depth_map.shape[2:]==image_size
    batch, _, height, width = depth_map.shape
    
    if from_ndc:
        h_start, h_end  = max(1, height/width), -max(1, height/width)
        w_start, w_end  = max(1, width/height), -max(1, width/height)
        h_step, w_step = -(h_start-h_end)/height, -(w_start-w_end)/width
    else:
        h_start, h_end = 0, height
        w_start, w_end = 0, width
        h_step, w_step = 1, 1
    grid_w, grid_h = torch.meshgrid(torch.arange(w_start, w_end, w_step), 
                                    torch.arange(h_start, h_end, h_step), indexing='xy')
    
    xy_map = torch.stack((grid_w, grid_h), dim=0)
    xy_map = torch.tile(xy_map, (batch,1,1,1)).to(depth_map.device)
    xy_depth = torch.cat([xy_map, depth_map], axis=1) # B x 3 x H x W
    
    return xy_depth


def get_RT_inverse(
    R: torch.Tensor,
    T: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    helper to derive inverse R,T transform from rotation (Nx3x3) and translation (Nx3x1)
    """
    assert T.shape[0]==R.shape[0], "Sequence length broken"
    assert T.shape[1:]==(3,1), "Translation transform has invalid shape"
    assert R.shape[1:]==(3,3), "Rotation transform has invalid shape"
    
    R_ = R.transpose(1,2)
    T_ = -torch.matmul(R_, T)
    
    return R_, T_


def get_pytorch3d_matrix(
    R: torch.Tensor,
    T: torch.Tensor
) -> torch.Tensor:
    """
    helper to derive Nx4x4 homogeneous transform from rotation (Nx3x3) and translation (Nx3x1)
    Output matrix follows pytorch3d matrix format
    NOTE: the rotation component is transposed
    M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]
    """
    assert T.shape[0]==R.shape[0], "Sequence length broken"
    assert T.shape[1:]==(3,1), "Translation transform has invalid shape"
    assert R.shape[1:]==(3,3), "Rotation transform has invalid shape"

    R_ = R.transpose(1,2)
    T_ = T.reshape(-1,1,3)

    M = torch.cat((R_, T_), dim=1) # N,4,3

    # Convert matrix into Nx4x4 homogeneous form
    M = torch.nn.functional.pad(M, (0,1,0,0,0,0), mode='constant', value=0)
    
    M[:,3,3] = 1

    return M


def label_to_one_hot(
    id_mask: np.ndarray,
    filter_void: bool=False,
    dtype: np.dtype=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.sort(np.unique(id_mask))
    bin_mask = np.zeros((len(ids),)+id_mask.shape, dtype=dtype)
    for i,idd in enumerate(ids):
        bin_mask[i] = id_mask==idd

    if filter_void and ids[0]==0:
        bin_mask = bin_mask[1:]
        ids = ids[1:]

    if bin_mask.ndim==4:
        bin_mask = np.transpose(bin_mask, axes=(1,0,2,3))

    return bin_mask, ids


def filter_idmask_area(
    id_mask: np.ndarray,
    filter_pcnt: float=0.001,
) -> np.ndarray:
    min_sum = id_mask.shape[-2]*id_mask.shape[-1]*filter_pcnt
    ids = np.sort(np.unique(id_mask))

    flt_mask = np.zeros(id_mask.shape, dtype=id_mask.dtype)
    for i,idd in enumerate(ids):
        bin_mask = id_mask==idd
        if bin_mask.sum()>=min_sum:
            flt_mask[bin_mask] = idd
    
    return flt_mask


def filter_binmask_area(
    bin_mask: np.ndarray,
    filter_pcnt: float=0.001,
) -> np.ndarray:
    min_sum = bin_mask.shape[-2]*bin_mask.shape[-1]*filter_pcnt

    bin_sum = np.sum(bin_mask, axis=(-2,-1))
    bin_sum = bin_sum >= min_sum
    if bin_mask.ndim==3:
        return bin_mask[bin_sum], bin_sum, min_sum
    elif bin_mask.ndim==4:
        bin_sum = np.any(bin_sum, axis=0)
        return bin_mask[:,bin_sum], bin_sum, min_sum
    else:
        raise ValueError

def colorize_depth(
    img: np.ndarray,
    cmap: str = 'viridis'
) -> None:
    cm = plt.get_cmap(cmap)
    return (cm(img) * 255).astype(np.uint8)[:,:,:3][:,:,::-1]


def make_video(
    video_dir: str,
    output_path: str
) -> None:
    files = [os.path.join(video_dir, fl) for fl in np.sort(os.listdir(video_dir))]
    first_image = cv2.imread(files[0])

    height, width, layers = first_image.shape  

    convert_to_mp4 = False
    if output_path.endswith(".mp4"):
        actual_output = output_path[:-4]+'.avi'
        convert_to_mp4 = True

    video = cv2.VideoWriter(actual_output, 0, fps=10, frameSize=(width, height))
  
    # Appending the images to the video one by one
    for img_file in files: 
        
        if 'imagesDEPTH' in img_file:
            import matplotlib
            matplotlib.use('TKAgg')
            image = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)/1000.0
            image /= image.max()
            image = colorize_depth(image)
        else:
            image = cv2.imread(img_file)

        video.write(image)
      
    cv2.destroyAllWindows() 
    video.release()

    if convert_to_mp4:
        import subprocess
        import shlex
        subprocess.run(shlex.split(f'ffmpeg -i {actual_output} -c:v libx264 -pix_fmt yuv420p -y {output_path}'))
        os.remove(actual_output)