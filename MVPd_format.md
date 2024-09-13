# MVPd Format

The Massive Video Panoptic dataset is formatted according to the [COCO panoptic segmentation format](https://cocodataset.org/#format-data) with slight modification to support videos (inspired by the definition defined by [VIPSeg (Miao et al.)](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset)) and videos with depth images to go with each RGB image. In addition, we append metadata to our annotation files to allow for each object instance to be related back to it's source mesh from the [Habitat-Matterport 3D Semantic Dataset (Yadav et al.)](https://aihabitat.org/datasets/hm3d-semantics/).

## Scene-Level Dataset Structure

At the highest level, MVPd is organized as a collection of training videos and validation videos. Within these two sets, videos are organized into compressed `.tar.gz` files, where each file corresponds to a single Matterport scan. For each scene, these `.tar.gz` files are a compressed represenation of the following directory structure:

```
└── <scene_ID>                    # HM3D Scene ID (e.g. `00006-HkseAnWCgqk`)
    ├── panoptic.json             # Annotation file
    │
    ├── panomasksRGB              # Directory containing panoptic labels
    │   └── {video_ID}            # Single video directory
    │       ├── {view_IDX}.png    # COCO panoptic api formatted label image
    │       │...
    │
    ├── imagesRGB  # Directory containing RGB images (ambient scene illumination by default)
    │   └── {video_ID}            # Single video directory
    │       ├── {view_IDX}.jpg    # JPEG formatted color image
    │       │...
    │
    └── imagesDEPTH               # Directory containing depth images for each video
        └── {video_ID}            # Single video directory
            └── {view_IDX}.png    # UINT16 formatted depth image in millimeters (factor of 1/1000 to convert to meters)
            ...
```

Within these scene-level datasets, each `panoptic.json` file includes annotations of the following format:

#### Scene-Level panoptic.json format

```
{
  "videos"        : [video],
  "annotations"   : [video_annotation],
  "categories"    : [license],
  "instances"     : [instance],
}


video{
  "video_id"       : str,
  "images"         : [image],
}

video_annotation{
  "video_id"        : str,
  "annotations"     : [annotation],
}


image{
  "id"              : int,
  "width"           : int,
  "height"          : int,
  "file_name"       : str,
  "depth_file_name" : str,
  "scene_id"        : str,
  "camera_position" : [float, float, float],        # Position(X, Y, Z)
  "camera_rotation" : [float, float, float, float], # Quaternion(W, X, Y, Z)
}

annotation{
  "image_id"        : int,
  "file_name"       : str,
  "segments_info"   : [segment_info],
}

segment_info{
  "id"              : int,
  "category_id"     : int,
  "area"            : int,
  "bbox"            : [x,y,width,height],
  "iscrowd"         : 0 or 1,
  "instance_id"     : int,
}

instance{
  "id"              : int,
  "category_id"     : int,
  "raw_category"    : str,
  "color"           : [R,G,B], # Color from Matterport source scene
  "scene_id"        : str,
}

category{
  "id"              : int,
  "name"            : str,
  "supercategory"   : str,
  "isthing"         : 0 or 1,
  "color"           : [R,G,B],
}


```

Notably, the `instance` metadata is included to allow for correlating every object instance in MVPd images with the corresponding scene and instance mesh of HM3D.



---


## Composed Dataset Structure

After extracting and merging scene-level datasets (using `scripts/preprocessing/format.sh`), each split of the dataset will have the same annotation file format as follows:

#### Annotation Format: panoptic_{train|val|test}.json

The training, validation and testing annotation files take the following format:

```
{
  "videos"        : [video],
  "annotations"   : [video_annotation],
  "categories"    : [license],
}


video{
  "video_id"       : str,
  "images"         : [image],
}

video_annotation{
  "video_id"        : str,
  "annotations"     : [annotation],
}


image{
  "id"              : int,
  "width"           : int,
  "height"          : int,
  "file_name"       : str,
  "depth_file_name" : str,
  "scene_id"        : str,
  "camera_position" : [float, float, float],        # Position(X, Y, Z)
  "camera_rotation" : [float, float, float, float], # Quaternion(W, X, Y, Z)
}

annotation{
  "image_id"        : int,
  "file_name"       : str,
  "segments_info"   : [segment_info],
}

segment_info{
  "id"              : int,
  "category_id"     : int,
  "area"            : int,
  "bbox"            : [x,y,width,height],
  "iscrowd"         : 0 or 1,
  "instance_id"     : int,
}

category{
  "id"              : int,
  "name"            : str,
  "supercategory"   : str,
  "isthing"         : 0 or 1,
  "color"           : [R,G,B],
}


```

Note that the `instance_id` field of each segment can be used to correlate every object instance from MVPd back to its corresponding source mesh in HM3D. For additional details on annotation format, kindly refer to [MVPd_format.md](MVPd_format.md).

