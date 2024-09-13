import os
import sys
import argparse
import configparser
import csv

import math
import numpy as np
import itertools, functools, operator


import bpy
import bmesh
from mathutils import Vector, Euler, Quaternion

from PIL import Image


##############################################################################
#                             BLENDER UTILITIES                              #
##############################################################################


def get_camera(pos, rot, name="Camera_Sample", rot_mode='ZXY', lens_unit='FOV', angle_x=69, clip_start=1e-2, clip_end=1000, scale=(1,1,1)):
    camera = bpy.data.cameras.get(name)
    if camera is None:
        camera = bpy.data.cameras.new(name)
    camera.lens_unit = lens_unit
    camera.angle_x = math.radians(angle_x) # Convert to radians for blender
    camera.clip_start = clip_start
    camera.clip_end = clip_end
    
    camera_sample_obj = bpy.data.objects.new(name, camera)
    camera_sample_obj.location = Vector(pos)
    camera_sample_obj.rotation_mode = rot_mode
    camera_sample_obj.rotation_euler = Euler(rot)
    camera_sample_obj.scale = scale

    return camera_sample_obj


def delete_object(obj):
    if obj is not None:
        bpy.ops.object.delete({"selected_objects": [obj]})


def delete_collection(collection):
    if collection is not None:
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


def add_object_to_collection(object, collection):
    for coll in object.users_collection:
        coll.objects.unlink(object)
    collection.objects.link(object)


def reset_blend():
    for obj in bpy.data.objects:
        delete_object(obj)

##############################################################################
#                         END OF BLENDER UTILITIES                           #
##############################################################################


def render_scene_semantics(SCENE_DIR, SCENE_VIEWS_FILE, SCENE_OUT_DIR, CONFIG, verbose=True):
    
    SCENE_NAME = SCENE_DIR.split('/')[-1]
    SCENE_FILE = SCENE_NAME.split('-')[1]+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME.split('-')[1]+'.semantic.glb'
    
    if verbose:
        print()
        print("********************")
        print(f"SAMPLING VIEWS FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()

    os.makedirs(SCENE_OUT_DIR, exist_ok=True)

    if verbose:
        print()
        print("********************")
        print("RESETTING SCENE")

    reset_blend()
    
    general_collection = bpy.context.scene.collection

    delete_object(bpy.data.objects.get("Camera"))
    camera_obj = get_camera(pos=(0,0,0), 
        rot=(0,0,math.pi), 
        name="Camera", 
        rot_mode='QUATERNION',
        lens_unit=CONFIG['blender.camera']['lens_unit'], # leave units as string
        angle_x=CONFIG['blender.camera'].getfloat('angle_x'), 
        clip_start=CONFIG['blender.camera'].getfloat('clip_start'), 
        clip_end=config['blender.camera'].getfloat('clip_end'))
    add_object_to_collection(camera_obj, general_collection)
    bpy.context.scene.camera = camera_obj

    semantic_building_collection = bpy.data.collections.get("Semantic_Building")
    delete_collection(semantic_building_collection)

    if verbose:
        print("DONE RESETTING SCENE")
        print("********************")
        print()




    if verbose:
        print()
        print("***********************")
        print("INITIALIZING SCENE")
    
    bpy.ops.import_scene.gltf(filepath=os.path.join(SCENE_DIR,SEMANTIC_SCENE_FILE))

    semantic_building_collection = bpy.data.collections.get("Semantic_Building")
    if semantic_building_collection is None:
        semantic_building_collection = bpy.data.collections.new("Semantic_Building")
        bpy.context.scene.collection.children.link(semantic_building_collection)
    
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if obj.type=="MESH":
            add_object_to_collection(obj, semantic_building_collection)
            for mat in obj.data.materials:
                mat.use_backface_culling = False
                if mat.node_tree:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE':
                            node.interpolation = 'Closest'
    
    semantic_building_collection.hide_render=False


    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display.shading.light = 'FLAT'
    bpy.context.scene.display.shading.color_type = 'TEXTURE'
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.display.render_aa = 'OFF'
    bpy.context.scene.view_settings.view_transform = 'Standard'

    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_x = CONFIG['blender.resolution'].getint('resolution_x') # width
    bpy.context.scene.render.resolution_y = CONFIG['blender.resolution'].getint('resolution_y') # height

    if verbose:
        print("DONE INITIALIZING SCENE")
        print("***********************")
        print()

        

    if verbose:
        print()
        print("***********************")
        print(f"INITIATING RENDERING")


    render_image_count = 0

    with open(SCENE_VIEWS_FILE, 'r') as csvfile:

        pose_reader = csv.reader(csvfile, delimiter=',')

        for pose_meta in pose_reader:
            info_ID = pose_meta[:4]
            info_position = pose_meta[4:7]
            info_rotation = pose_meta[7:11]

            # Skip information line if it is first
            if info_ID[0]=='Scene-ID':
                continue

            x_pos, y_pos, z_pos = info_position
            quat_w, quat_x, quat_y, quat_z = info_rotation

            # Parse pose infomration out of string type
            x_pos, y_pos, z_pos = float(x_pos), float(y_pos), float(z_pos)
            quat_w, quat_x, quat_y, quat_z = float(quat_w), float(quat_x), float(quat_y), float(quat_z)

            render_out_path =  os.path.join(SCENE_OUT_DIR, f"{'.'.join(info_ID)}.SEM.png")
            if os.path.isfile(render_out_path):
                continue

            # Set camera position
            camera_obj.location = Vector((x_pos, y_pos, z_pos))

            # Set camera rotation
            camera_obj.rotation_quaternion = Quaternion((quat_w, quat_x, quat_y, quat_z))

            # Update scene view layer to recalculate camera extrensic matrix
            bpy.context.view_layer.update()

            bpy.context.scene.render.filepath = render_out_path
            bpy.ops.render.render(write_still = True)

            render_image_count += 1
    
    if verbose:
        print(f"DONE RENDERING {render_image_count} VIEWS")
        print("***********************")
        print()



if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='sample_views',
                    usage='blender --background --python <path to render_semantics.py> -- [options]',
                    description='Blender python script for rendering instance semantic segmentation label images from the Matterport 3D semantic dataset assuming valid views have already been sampled',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-config', '--config-file', help='path to ini file containing rendering and sampling configuration', type=str)
    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args(argv)

    config = configparser.ConfigParser()
    config.read(args.config_file)

    if args.verbose:
        print()
        print(args)
        print()
        print(config)
        print()

    scene_directories = sorted([path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))])
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)

        scene_files = os.listdir(scene_dir_path)

        scene_out_path = os.path.join(args.output_dir, scene_dir)
        scene_view_poses_path = os.path.join(scene_out_path, scene_dir+'.render_view_poses.csv')

        scene_has_semantic_mesh = any([fl.endswith('.semantic.glb') for fl in scene_files])
        scene_has_semantic_txt = any([fl.endswith('.semantic.txt') for fl in scene_files])
        scene_has_sampled_views = os.path.isfile(scene_view_poses_path)

        if scene_has_semantic_mesh and scene_has_semantic_txt and scene_has_sampled_views:
            render_scene_semantics(scene_dir_path, scene_view_poses_path, scene_out_path, config, verbose=args.verbose)


    if args.verbose:
        print()
        print("***********************")
        print(f"DONE RENDERING ALL SCENES")
        print("***********************")
        print()