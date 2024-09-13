# Blender Simulation Workflow


This folder contains all blender python scripts for rendering visual imagery and corresponding instance segmentation labels from the [Matterport 3D Semantic dataset](https://aihabitat.org/datasets/hm3d-semantics/).

## Prerequisites

1. Install [Blender LTS](https://www.blender.org/download/releases/3-3/). This code was developed and tested with Blenderv3.3.7 on Ubuntu 20.04.
    - Details in root [README.md](../../README.md#blender)
2. Download and extract all scenes in the [Matterport 3D Semantic dataset](https://aihabitat.org/datasets/hm3d-semantics/)
    - Details in root [README.md](../../README.md#habitat-matterport-3d-semantic-dataset)


## Simulation Workflow

1. **Calculate valid camera views**
    - Run a blender command line script (i.e. ./blender.exe --python script.py -- command line args) to sample valid views for each scene in the matterport dataset
    - Input: desired sampling resolution in position and rotation and camera parameters (focal length, image size)
    - Output: one csv file per scene listing the valid camera poses
        - Valid camera pose defined as one with all corners and center ray viewing inner mesh and at least 0.25 meters from camera origin
        - Changing this definition will require re-rendering
    - **Shortcut**
        ```
        ./simulators/blender/blender-3.3.7-linux-x64/blender \
          --python ./simulators/blender/sample_views.py \
            -- \
            -config ./simulators/blender/configs/render_config.ini \
            -data ./datasets/HM3D/example/ \
            -out ./datasets/renders/example/
      ```
    - **Shortcut with Habitat-Sim Trajectories**
      ```
      ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/blender \
        --python ./zero_shot_scene_segmentation/simulators/blender/sample_views.py \
          -- \
          -config simulators/blender/configs/render_config.ini \
          -data datasets/raw_data/HM3D/example/ \
          -out datasets/raw_data/trajectory_renders/example/ \
          -habitat
      ```
    - **Visualize Sampled Views**
      ```
      ./simulators/blender/blender-3.3.7-linux-x64/blender \
        --python simulators/blender/visualize_views.py \
          -- \
          -scene 00861-GLAQ4DNUx5U \
          -config simulators/blender/configs/render_config.ini \
          -data datasets/HM3D/example/ \
          -out datasets/renders/example/
      ```
2. **Render semantic label images**
    - Run a blender command line script to render semantic images given valid view metadata
    - Input: meta data generated from step 1
    - Output: one directory per scene containing semantic images in png format which are mappable by name to corresponding pose
    - **Shortcut**
        ```
         ./simulators/blender/blender-3.3.7-linux-x64/blender \
           --background \
           --python simulators/blender/render_semantics.py \
             -- \
             -config simulators/blender/configs/render_config.ini \
             -data datasets/raw_data/HM3D/example/ \
             -out datasets/raw_data/trajectory_renders/example/
        ```

4. Render visual images
    - Run a blender command line script to render visual RGB images given valid view metadata
    - Input: meta data generated from step 1 and lighting parameters (i.e. spot light strength)
    - Output: one directory per scene containing semantic images in png format which are mappable by name to corresponding pose and light conditions
    - **Shortcut**
        ````
        ./simulators/blender/blender-3.3.7-linux-x64/blender \
          --background \
          --python simulators/blender/render_color.py \
            -- \
            -config simulators/blender/configs/render_config.ini \
            -light-config simulators/blender/configs/illumination_0000000000_config.ini \
            -data datasets/HM3D/example/ \
            -out datasets/renders/example/
        ````


## Simulation Debugging

The blender python script in [workbench.py](workbench.py)` is intended for use in the interactive Blender GUI. This script allows visualization of the rejection sampling process used for view sampling and can be used for interactive script development.

