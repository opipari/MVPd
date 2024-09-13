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



##############################################################################
#                             BLENDER UTILITIES                              #
##############################################################################

def get_sphere(pos, rot, name='Basic_Sphere', mat=None):
    sphere_mesh = bpy.data.meshes.get(name)
    if sphere_mesh is None:
        sphere_mesh = bpy.data.meshes.new(name)
        
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.1)
        bm.to_mesh(sphere_mesh)
        bm.free()
        
        
    sphere_obj = bpy.data.objects.new(name, sphere_mesh)
    sphere_obj.location = Vector(pos)
    sphere_obj.rotation_mode = 'ZXY'
    sphere_obj.rotation_quaternion = Quaternion(rot)
    sphere_obj.scale = Vector((1,1,1))
    if mat is not None:
        sphere_obj.data.materials.append(mat)
    
    bpy.context.collection.objects.link(sphere_obj)
    
#    bpy.ops.object.select_all(action='DESELECT')
#    bpy.context.view_layer.objects.active = sphere_obj
#    sphere_obj.select_set(True)
#    bpy.ops.object.modifier_add(type='SUBSURF')
#    bpy.ops.object.shade_smooth()
    
    return sphere_obj

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
    camera_sample_obj.rotation_quaternion = Quaternion(rot)
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


def visualize_trajectory_samples(SCENE_DIR, SCENE_ALL_VIEWS_FILE, CONFIG, traj_idx, height, verbose=True):
    
    SCENE_NAME = SCENE_DIR.split('/')[-1]
    SCENE_FILE = SCENE_NAME.split('-')[1]+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME.split('-')[1]+'.semantic.glb'

    if verbose:
        print()
        print("********************")
        print(f"POST PROCESSING SEMANTICS FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()



    if verbose:
        print()
        print("********************")
        print("RESETTING SCENE")

    initial_sample_collection = bpy.data.collections.get("Initial_Sample_Grid")
    delete_collection(initial_sample_collection)

    accepted_sample_collection = bpy.data.collections.get("Accepted_Sample_Grid")
    delete_collection(accepted_sample_collection)

    building_collection = bpy.data.collections.get("Building")
    delete_collection(building_collection)

    semantic_building_collection = bpy.data.collections.get("Semantic_Building")
    delete_collection(semantic_building_collection)

    bpy.ops.wm.read_homefile()
    reset_blend()

    general_collection = bpy.context.scene.collection

    if verbose:
        print("DONE RESETTING SCENE")
        print("********************")
        print()




    if verbose:
        print()
        print("***********************")
        print("INITIALIZING SCENE")

    general_collection = bpy.context.scene.collection
    
    bpy.ops.import_scene.gltf(filepath=os.path.join(SCENE_DIR,SCENE_FILE))

    building_collection = bpy.data.collections.get("Building")
    if building_collection is None:
        building_collection = bpy.data.collections.new("Building")
        bpy.context.scene.collection.children.link(building_collection)
    
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if obj.type=="MESH":
            add_object_to_collection(obj, building_collection)
            for mat in obj.data.materials:
                mat.use_backface_culling = True
                if mat.node_tree:
                    node_tree = mat.node_tree
                    node_types = [node.type for node in node_tree.nodes]
                    
                    assert set(['TEX_IMAGE','BSDF_PRINCIPLED','OUTPUT_MATERIAL'])==set(node_types)

                    tex_node = node_tree.nodes['Image Texture']
                    bsdf_node = node_tree.nodes['Principled BSDF']
                    mat_node = node_tree.nodes['Material Output']


                    bsdf_node.inputs["Specular"].default_value = CONFIG['blender.BSDF_PRINCIPLED'].getfloat('Specular')
                    bsdf_node.inputs["Roughness"].default_value = CONFIG['blender.BSDF_PRINCIPLED'].getfloat('Roughness')
    
    bpy.ops.import_scene.gltf(filepath=os.path.join(SCENE_DIR,SEMANTIC_SCENE_FILE))

    semantic_building_collection = bpy.data.collections.get("Semantic_Building")
    if semantic_building_collection is None:
        semantic_building_collection = bpy.data.collections.new("Semantic_Building")
        bpy.context.scene.collection.children.link(semantic_building_collection)
    
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if obj.type=="MESH" and building_collection not in obj.users_collection:
            add_object_to_collection(obj, semantic_building_collection)
            for mat in obj.data.materials:
                mat.use_backface_culling = True
                if mat.node_tree:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE':
                            node.interpolation = 'Closest'

    # Create collections to house sample objects
    initial_sample_collection = bpy.data.collections.get("Initial_Sample_Grid")
    if initial_sample_collection is None:
        initial_sample_collection = bpy.data.collections.new("Initial_Sample_Grid")
        general_collection.children.link(initial_sample_collection)


    accepted_sample_collection = bpy.data.collections.get("Accepted_Sample_Grid")
    if accepted_sample_collection is None:
        accepted_sample_collection = bpy.data.collections.new("Accepted_Sample_Grid")
        general_collection.children.link(accepted_sample_collection)


    # Define shared material for spheres to represent position samples
    sphere_mat = bpy.data.materials.get("Sphere_Material")
    if sphere_mat is None:
        sphere_mat = bpy.data.materials.new(name="Sphere_Material")

    delete_object(bpy.data.objects.get("Camera"))
    camera_obj = get_camera(pos=(0,0,0), 
        rot=(0,0,math.pi), 
        name="Camera", 
        rot_mode='ZXY',
        lens_unit=CONFIG['blender.camera']['lens_unit'], # leave units as string
        angle_x=CONFIG['blender.camera'].getfloat('angle_x'), 
        clip_start=CONFIG['blender.camera'].getfloat('clip_start'), 
        clip_end=CONFIG['blender.camera'].getfloat('clip_end'))
    add_object_to_collection(camera_obj, general_collection)
    bpy.context.scene.camera = camera_obj

    delete_object(bpy.data.objects.get("Spot_Light"))
    
    spot_light = bpy.data.lights.new(name="Spot_Light", type='SPOT')
    spot_light.energy = 50
    spot_light.spot_size = math.radians(180)
    spot_light.spot_blend = 1.0
    spot_light.distance = 25
    spot_light.shadow_soft_size = 0.0
    spot_light_obj = bpy.data.objects.new(name="Spot_Light", object_data=spot_light)
    spot_light_obj.location = Vector((0,0,0))
    spot_light_obj.parent = camera_obj
    add_object_to_collection(spot_light_obj, general_collection)


    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_x = CONFIG['blender.resolution'].getint('resolution_x') # width
    bpy.context.scene.render.resolution_y = CONFIG['blender.resolution'].getint('resolution_y') # height

    if verbose:
        print("DONE INITIALIZING SCENE")
        print("***********************")
        print()

    
    visualized_pos_samples = []

    with open(SCENE_ALL_VIEWS_FILE, 'r') as csvfile:

        pose_reader = csv.reader(csvfile, delimiter=',')

        for pose_meta in pose_reader:
            info_ID = pose_meta[:4]
            info_position = pose_meta[4:7]
            info_rotation = pose_meta[7:11]

            # Skip information line if it is first
            if info_ID[0]=='Scene-ID':
                continue

            print(info_ID[1], info_ID[2])
            if int(info_ID[1])!=traj_idx or int(info_ID[2])!=height:
                continue

            x_pos, y_pos, z_pos = info_position
            quat_w, quat_x, quat_y, quat_z = info_rotation

            # Parse pose infomration out of string type
            x_pos, y_pos, z_pos = float(x_pos), float(y_pos), float(z_pos)
            quat_w, quat_x, quat_y, quat_z = float(quat_w), float(quat_x), float(quat_y), float(quat_z)


            camera_sample = get_camera(pos=(x_pos, y_pos, z_pos), 
                rot=(quat_w, quat_x, quat_y, quat_z), 
                name="View-ID-"+str(info_ID[-1]), 
                rot_mode='QUATERNION',
                lens_unit=CONFIG['blender.camera']['lens_unit'], # leave units as string
                angle_x=CONFIG['blender.camera'].getfloat('angle_x'), 
                clip_start=CONFIG['blender.camera'].getfloat('clip_start'), 
                clip_end=config['blender.camera'].getfloat('clip_end'))
            add_object_to_collection(camera_sample, accepted_sample_collection)



    
            
            
            

    if verbose:
        print()
        print("********************")
        print(f"DONE POST PROCESSING SEMANTICS FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()


if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='sample_views',
                    usage='blender --python <path to visualize_views.py> -- [options]',
                    description='Blender python script for visualizing sampled views from Matterport dataset assuming valid views have already been sampled',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-scene', '--scene-dir', help='name of folder containing rendered samples from Matterport dataset', type=str)
    parser.add_argument('-config', '--config-file', help='path to ini file containing rendering and sampling configuration', type=str)
    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-traj_idx', '--trajectory-index', type=int, default=0)
    parser.add_argument('-height', '--sensor-height', type=int, default=100)
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

    
    scene_dir_path = os.path.join(args.dataset_dir, args.scene_dir)

    scene_all_view_poses_path = os.path.join(args.output_dir, args.scene_dir, args.scene_dir+'.render_view_poses.csv')

    scene_has_sampled_views = os.path.isfile(scene_all_view_poses_path)

    if scene_has_sampled_views:
        visualize_trajectory_samples(scene_dir_path, scene_all_view_poses_path, config, args.trajectory_index, args.sensor_height, verbose=args.verbose)

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE CREATING VISUALIZATION")
        print("***********************")
        print()