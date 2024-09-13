import os
import sys
import csv
import argparse
import configparser

import math
import numpy as np
import itertools, functools, operator


import bpy
import bmesh
from mathutils import Vector, Euler, Quaternion

import cv2




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
    sphere_obj.rotation_euler = Euler(rot)
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






##############################################################################
#                             SPATIAL UTILITIES                              #
##############################################################################


def euclidean_distance(v1, v2):
    """Calculate euclidean distance between vectors.

    Keyword arguments:
    v1 -- 3D Vector
    v2 -- 3D Vector
    """
    diff = v1 - v2
    return math.sqrt(diff.x**2 + diff.y**2 + diff.z**2)


def bounding_box(ob_name, coords, edges=[], faces=[]):
    """Create mesh object representing object bounding boxes.

    Keyword arguments:
    ob_name -- new object name
    coords -- float triplets eg: [(-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)]
    edges -- int pairs eg: [(0,1), (0,2)]
    """

    # Create new mesh and a new object
    me = bpy.data.meshes.new(ob_name + "Mesh")
    ob = bpy.data.objects.new(ob_name, me)

    # Make a mesh from a list of vertices/edges/faces
    me.from_pydata(coords, edges, faces)

    # Display name and update the mesh
    ob.show_name = True
    me.update()
    return ob


def get_mesh_aabb(mesh_obj):
    """Calculate axis-aligned bounding box around mesh object in world frame.

    Keyword arguments:
    mesh_obj -- mesh object in blender scene
    """
    
    # Transform local corners to world coordinate frame
    bbox_corners = np.array([mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box])
    
    # Calculate bounding box in axis-aligned format
    bbox_min = np.min(bbox_corners,axis=0)
    bbox_max = np.max(bbox_corners,axis=0)
    
    bound_x = [bbox_min[0],bbox_max[0]]
    bound_y = [bbox_min[1],bbox_max[1]]
    bound_z = [bbox_min[2],bbox_max[2]]
    
    # Return corners of boundning box
    aabb_corners = [Vector((x,y,z)) for x,y,z in itertools.product(bound_x, bound_y, bound_z)]
    
    return aabb_corners



def aabb_3d_inter_areas(aabb1, aabb2):
    """Calculate 3D IOU from two axis-aligned bounding boxes.

    Keyword arguments:
    aabb1 -- list of float triplets representing first box corners
    aabb2 -- list of float triplets representing second box corners
    """
    aabb1, aabb2 = np.array(aabb1), np.array(aabb2)
    
    aabb1_min, aabb1_max = np.min(aabb1,axis=0), np.max(aabb1,axis=0)
    aabb2_min, aabb2_max = np.min(aabb2,axis=0), np.max(aabb2,axis=0)
    
    aabb1_area = np.prod(aabb1_max - aabb1_min)
    aabb2_area = np.prod(aabb2_max - aabb2_min)
    
    if np.any(aabb1_min>aabb2_max) or np.any(aabb1_max<aabb2_min):
        return 0.0, aabb1_area, aabb2_area
        
    inter_min = np.maximum(aabb1_min, aabb2_min)
    inter_max = np.minimum(aabb1_max, aabb2_max)
    
    inter_area = np.prod(inter_max - inter_min)

    iou = inter_area / aabb2_area#(aabb1_area + aabb2_area - inter_area)
    
    assert iou>=0.0 and iou<=1.0
    return inter_area, aabb1_area, aabb2_area

def merge_aabb(aabb1, aabb2):
    """Combine two axis-algined bounding boxes into one.

    Keyword arguments:
    aabb1 -- list of float triplets representing first box corners
    aabb2 -- list of float triplets representing second box corners
    """
    aabb1, aabb2 = np.array(aabb1), np.array(aabb2)
    
    aabb1_min, aabb1_max = np.min(aabb1,axis=0), np.max(aabb1,axis=0)
    aabb2_min, aabb2_max = np.min(aabb2,axis=0), np.max(aabb2,axis=0)
    
    aabb_min = np.minimum(aabb1_min, aabb2_min)
    aabb_max = np.maximum(aabb1_max, aabb2_max)
    
    bound_x = [aabb_min[0],aabb_max[0]]
    bound_y = [aabb_min[1],aabb_max[1]]
    bound_z = [aabb_min[2],aabb_max[2]]
    
    aabb_corners = [Vector((x,y,z)) for x,y,z in itertools.product(bound_x, bound_y, bound_z)]
    
    return aabb_corners

        
def get_collection_aabb(collection):
    """Calculate axis-aligned bounding box around collection of meshes.

    Keyword arguments:
    collection -- blender collection object
    """
    assert len(collection.all_objects)>0
    
    aabb = get_mesh_aabb(collection.all_objects[0])
    for i, obj_i in enumerate(collection.all_objects):
        aabb_i = get_mesh_aabb(obj_i)
        aabb = merge_aabb(aabb, aabb_i)
    return aabb


##############################################################################
#                         END OF SPATIAL UTILITIES                           #
##############################################################################





##############################################################################
#                            SAMPLING UTILITIES                              #
##############################################################################


def get_grid_points(aabb_bounds, samples_per_meter=1, margin_pcnt=0.025):
    """Calculate uniform grid of 3D location samples.

    Keyword arguments:
    aabb_bounds -- list of float tripliets representing corners of 3D boundary for samples
    samples_per_meter -- float for density of samples in each dimension
    margin_pcnt -- float representing inner margin on sampling bounds as percent of smallest boundary dimension
    """
    
    aabb_bounds = np.array(aabb_bounds)
    
    bounds_min = np.min(aabb_bounds, axis=0)
    bounds_max = np.max(aabb_bounds, axis=0)
    bounds_range = bounds_max-bounds_min
    
    margin = margin_pcnt * np.min(bounds_range)
    
    bounds_min += margin
    bounds_max -= margin
    bounds_range -= 2 * margin
    
    num_samples = np.floor(bounds_range * samples_per_meter).astype(dtype=np.int32)
    
    return np.linspace(bounds_min[0], bounds_max[0], num=num_samples[0], endpoint=True), \
            np.linspace(bounds_min[1], bounds_max[1], num=num_samples[1], endpoint=True), \
            np.linspace(bounds_min[2], bounds_max[2], num=num_samples[2], endpoint=True)


def get_grid_euler(roll_bounds=(math.radians(145),math.radians(225)), roll_samples=3, 
                   pitch_bounds=(math.radians(-20),math.radians(20)), pitch_samples=3, 
                   yaw_bounds=(0,2*math.pi-(2*math.pi/8)), yaw_samples=8):
    """Calculate uniform grid of samples in euler angle rotations.

    Keyword arguments:
    roll_bounds -- float pairs of min and max roll angles as radians
    roll_samples -- samples per degree in roll (Z)
    
    pitch_bounds -- float pairs of min and max roll angles as radians
    pitch_samples -- samples per degree in pitch (X)
    
    yaw_bounds -- float pairs of min and max roll angles as radians
    yaw_samples -- samples per degree in yaw (Y)
    """
    
    return np.linspace(roll_bounds[0], roll_bounds[1], num=roll_samples, endpoint=True), \
            np.linspace(pitch_bounds[0], pitch_bounds[1], num=pitch_samples, endpoint=True), \
            np.linspace(yaw_bounds[0], yaw_bounds[1], num=yaw_samples, endpoint=True)
            

def camera_viewing_valid_surface(camera_obj, dist_threshold=0.25):
    camera_center_and_corner_dirs = [((camera_obj.matrix_world @ corner) - camera_obj.location).normalized() for corner in [Vector((0,0,-1))]+list(camera_obj.data.view_frame(scene=bpy.context.scene))]
    
    camera_center_and_corner_ray_casts = [bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, camera_obj.location, camera_dir) for camera_dir in camera_center_and_corner_dirs]
    camera_center_and_corner_normal_dots = [corner_dir.dot(ray_cast[2].normalized()) for (corner_dir, ray_cast) in zip(camera_center_and_corner_dirs, camera_center_and_corner_ray_casts)]
    #camera_center_and_corner_ray_dists = [euclidean_distance(ray_cast[1], camera_obj.location) for ray_cast in camera_center_and_corner_ray_casts]
    camera_center_and_corner_depth = [-1*(camera_obj.matrix_world.inverted() @ ray_cast[1]).z for ray_cast in camera_center_and_corner_ray_casts]

    # Valid view if all corners and center ray hit inner mesh of scene
    all_rays_hit_surface = all([ray_cast[0] for ray_cast in camera_center_and_corner_ray_casts])
    all_rays_hit_inner_mesh = all([dot<0 for dot in camera_center_and_corner_normal_dots])
    all_rays_hit_within_dist_threshold = all([dist>=dist_threshold for dist in camera_center_and_corner_depth])
    
    return all_rays_hit_surface and all_rays_hit_inner_mesh and all_rays_hit_within_dist_threshold

def render_depth():
    # Render image for depth map
    bpy.ops.render.render()
    render_image = bpy.data.images['Viewer Node']
    image_width, image_height = render_image.size[:]

    depth_arr = np.array(render_image.pixels[:]).reshape((image_height, image_width, 4))[:,:,0]
    min_depth = depth_arr.min()
    
    depth_arr[depth_arr>=bpy.context.scene.camera.data.clip_end] = 0

    return min_depth, depth_arr

##############################################################################
#                         END OF SAMPLING UTILITIES                          #
##############################################################################


            
def sample_scene_views(SCENE_DIR, OUT_DIR, CONFIG, verbose=True):
    
    SCENE_NAME = SCENE_DIR.split('/')[-1]
    SCENE_FILE = SCENE_NAME.split('-')[1]+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME.split('-')[1]+'.semantic.glb'
    SCENE_OUT_DIR = os.path.join(OUT_DIR, SCENE_NAME)

    if verbose:
        print()
        print("********************")
        print(f"SAMPLING VIEWS FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SCENE_OUT_DIR, exist_ok=True)

    if verbose:
        print()
        print("********************")
        print("RESETTING SCENE")

    bpy.ops.wm.read_homefile()
    reset_blend()
    
    general_collection = bpy.context.scene.collection

    building_collection = bpy.data.collections.get("Building")
    delete_collection(building_collection)

    if verbose:
        print("DONE RESETTING SCENE")
        print("********************")
        print()



    if verbose:
        print()
        print("***********************")
        print("INITIALIZING SCENE")
    
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
                mat.use_backface_culling = False
    

    delete_object(bpy.data.objects.get("Camera"))
    camera_obj = get_camera(pos=(0,0,0), 
        rot=(0,0,math.pi), 
        name="Camera", 
        rot_mode='ZXY',
        lens_unit=CONFIG['blender.camera']['lens_unit'], # leave units as string
        angle_x=CONFIG['blender.camera'].getfloat('angle_x'), 
        clip_start=CONFIG['blender.camera'].getfloat('clip_start'), 
        clip_end=config['blender.camera'].getfloat('clip_end'))
    add_object_to_collection(camera_obj, general_collection)
    bpy.context.scene.camera = camera_obj


    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.eevee.taa_samples = 1
    bpy.context.scene.eevee.sss_samples = 1
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_x = CONFIG['blender.resolution'].getint('resolution_x') # width
    bpy.context.scene.render.resolution_y = CONFIG['blender.resolution'].getint('resolution_y') # height
    

    # Add z pass to view layer for rendering depth
    bpy.context.view_layer.use_pass_z = True

    # Use compositing nodes for rendering depth
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree

    # Remove default nodes
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)


    render_layers = node_tree.nodes.new('CompositorNodeRLayers')
    node_viewer = node_tree.nodes.new('CompositorNodeViewer')
    node_viewer.use_alpha = False
    node_viewer.select = True
    bpy.context.scene.node_tree.nodes.active = node_viewer

    # Link depth output from z_pass to node viewier Image input
    node_tree.links.new(render_layers.outputs['Depth'], node_viewer.inputs['Image']) # link Z to output


    if verbose:
        print("DONE INITIALIZING SCENE")
        print("***********************")
        print()


    building_aabb = get_collection_aabb(building_collection)

    grid_x, grid_y, grid_z = get_grid_points(building_aabb, samples_per_meter=CONFIG['view_sampling'].getfloat('position_samples_per_meter'))
    num_pos_samples = functools.reduce(operator.mul, map(len, (grid_x, grid_y, grid_z)), 1)

    grid_roll, grid_pitch, grid_yaw = get_grid_euler(roll_bounds=(math.radians(CONFIG['view_sampling'].getfloat('roll_samples_minimum')),
                                                                  math.radians(CONFIG['view_sampling'].getfloat('roll_samples_maximum'))
                                                                  ), 
                                                        roll_samples=CONFIG['view_sampling'].getint('roll_samples_count'), 
                                                        pitch_bounds=(math.radians(CONFIG['view_sampling'].getfloat('pitch_samples_minimum')), 
                                                                      math.radians(CONFIG['view_sampling'].getfloat('pitch_samples_maximum'))
                                                                      ), 
                                                        pitch_samples=CONFIG['view_sampling'].getint('pitch_samples_count'), 
                                                        yaw_bounds=(math.radians(CONFIG['view_sampling'].getfloat('yaw_samples_minimum')), 
                                                                    math.radians(CONFIG['view_sampling'].getfloat('yaw_samples_maximum'))
                                                                    ), 
                                                        yaw_samples=CONFIG['view_sampling'].getint('yaw_samples_count')
                                                        )
    num_rot_samples = functools.reduce(operator.mul, map(len, (grid_roll, grid_pitch, grid_yaw)), 1)
    


    
    all_view_file = open(os.path.join(SCENE_OUT_DIR, f"{SCENE_NAME}.all_view_poses.csv"), "w")
    all_view_file.write('Scene-ID,View-ID,Position-ID,Rotation-ID,X-Position,Y-Position,Z-Position,Roll-Z-EulerZXY,Pitch-X-EulerZXY,Yaw-Y-EulerZXY,Accepted-Y-N\n')
    all_view_file.flush()


    accepted_view_file = open(os.path.join(SCENE_OUT_DIR, f"{SCENE_NAME}.render_view_poses.csv"), "w")
    accepted_view_file.write('Scene-ID,Valid-View-ID,Position-ID,Rotation-ID,X-Position,Y-Position,Z-Position,W-Quaternion,X-Quaternion,Y-Quaternion,Z-Quaternion\n')
    accepted_view_file.flush()
    
    if verbose:
        print()
        print("***********************")
        print(f"INITIATING SIMULATION OF {num_pos_samples*num_rot_samples} VIEW SAMPLES")


    valid_view_count = 0

    # Iterate over uniform grid of positions within scene bounding box
    for pos_i, (x,y,z) in enumerate(itertools.product(grid_x, grid_y, grid_z)):
        
        # Set camera position
        camera_obj.location = Vector((x,y,z))
        
        # Iterate over uniform grid of rotations
        for rot_i,(roll,pitch,yaw) in enumerate(itertools.product(grid_roll, grid_pitch, grid_yaw)):

            # Set camera rotation
            camera_obj.rotation_euler = Euler((pitch,yaw,roll))

            # Update scene view layer to recalculate camera extrensic matrix
            bpy.context.view_layer.update()

            is_valid_view = False
            
            # Determine if rejection sampling criteria is met
            # Valid view defined as one where corner and center rays view inner mesh surface and at least 0.25m from camera origin
            if camera_viewing_valid_surface(camera_obj, dist_threshold=CONFIG['view_sampling'].getfloat('surface_distance_threshold')):

                min_depth, depth_arr = render_depth()
                if min_depth>=CONFIG['view_sampling'].getfloat('surface_distance_threshold'):

                    depth_arr = np.flipud(np.round(depth_arr*1000).astype(np.uint16))
                    cv2.imwrite(os.path.join(SCENE_OUT_DIR, f'{SCENE_NAME}.{valid_view_count:010}.{pos_i:010}.{rot_i:010}.DEPTH.png'), depth_arr)


                    camera_quaternion = camera_obj.rotation_euler.to_quaternion()
                    quat_w, quat_x, quat_y, quat_z = camera_quaternion.w, camera_quaternion.x, camera_quaternion.y, camera_quaternion.z

                    all_view_file.write(f'{SCENE_NAME},{(pos_i*num_rot_samples)+rot_i:010},{pos_i:010},{rot_i:010},{x},{y},{z},{roll},{pitch},{yaw},Y\n')
                    all_view_file.flush()

                    accepted_view_file.write(f'{SCENE_NAME},{valid_view_count:010},{pos_i:010},{rot_i:010},{x},{y},{z},{quat_w},{quat_x},{quat_y},{quat_z}\n')
                    accepted_view_file.flush()

                    valid_view_count += 1
                    is_valid_view = True

            if not is_valid_view:
                all_view_file.write(f'{SCENE_NAME},{(pos_i*num_rot_samples)+rot_i:010},{pos_i:010},{rot_i:010},{x},{y},{z},{roll},{pitch},{yaw},N\n')
                all_view_file.flush()

            if rot_i%100==0 and rot_i!=0:
                print(f'    {rot_i}/{num_rot_samples} rotation samples finished simulating')
            
        if pos_i%100==0 and pos_i!=0:
            print(f'{pos_i}/{num_pos_samples} position samples finished simulating')
    

    all_view_file.close()
    accepted_view_file.close()

    if verbose:
        print("***********************")
        print(f"DONE SIMULATING {num_pos_samples*num_rot_samples} VIEW SAMPLES")
        print(f"ACCEPTED {valid_view_count} VALID VIEWS")
        print("***********************")
        print()




def sample_scene_trajectories(SCENE_DIR, OUT_DIR, CONFIG, verbose=True):


    SCENE_NAME = SCENE_DIR.split('/')[-1]
    SCENE_FILE = SCENE_NAME.split('-')[1]+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME.split('-')[1]+'.semantic.glb'
    SCENE_OUT_DIR = os.path.join(OUT_DIR, SCENE_NAME)
    SCENE_TRAJECTORIES_FILE = SCENE_NAME+'.habitat_trajectory_poses.csv'

    if verbose:
        print()
        print("********************")
        print(f"SAMPLING VIEWS FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SCENE_OUT_DIR, exist_ok=True)

    if verbose:
        print()
        print("********************")
        print("RESETTING SCENE")

    bpy.ops.wm.read_homefile()
    reset_blend()
    
    general_collection = bpy.context.scene.collection

    building_collection = bpy.data.collections.get("Building")
    delete_collection(building_collection)

    if verbose:
        print("DONE RESETTING SCENE")
        print("********************")
        print()



    if verbose:
        print()
        print("***********************")
        print("INITIALIZING SCENE")
    
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
                mat.use_backface_culling = False
    

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


    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.eevee.taa_samples = 1
    bpy.context.scene.eevee.sss_samples = 1
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_x = CONFIG['blender.resolution'].getint('resolution_x') # width
    bpy.context.scene.render.resolution_y = CONFIG['blender.resolution'].getint('resolution_y') # height
    

    # Add z pass to view layer for rendering depth
    bpy.context.view_layer.use_pass_z = True

    # Use compositing nodes for rendering depth
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree

    # Remove default nodes
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)


    render_layers = node_tree.nodes.new('CompositorNodeRLayers')
    node_viewer = node_tree.nodes.new('CompositorNodeViewer')
    node_viewer.use_alpha = False
    node_viewer.select = True
    bpy.context.scene.node_tree.nodes.active = node_viewer

    # Link depth output from z_pass to node viewier Image input
    node_tree.links.new(render_layers.outputs['Depth'], node_viewer.inputs['Image']) # link Z to output


    if verbose:
        print("DONE INITIALIZING SCENE")
        print("***********************")
        print()


    accepted_view_file = open(os.path.join(SCENE_OUT_DIR, f"{SCENE_NAME}.render_view_poses.csv"), "w")
    accepted_view_file.write('Scene-ID,Trajectory-ID,Sensor-Height-ID,View-ID,X-Position,Y-Position,Z-Position,W-Quaternion,X-Quaternion,Y-Quaternion,Z-Quaternion,View-Corners-On-Surface-Y-N,Minimum-Depth-Below-Threshold-Y-N,Is-Valid-View-Y-N\n')
    accepted_view_file.flush()
    
    if verbose:
        print()
        print("***********************")
        print(f"INITIATING SIMULATION OF TRAJECTORY SAMPLES")


    with open(os.path.join(SCENE_OUT_DIR, SCENE_TRAJECTORIES_FILE), 'r') as csvfile:
        num_views = sum(1 for row in csvfile)-1

    view_count = 0
    with open(os.path.join(SCENE_OUT_DIR, SCENE_TRAJECTORIES_FILE), 'r') as csvfile:

        pose_reader = csv.reader(csvfile, delimiter=',')

        for pose_meta in pose_reader:
            scene_name, traj_idx, sensor_height, view_idx, x_pos, y_pos, z_pos, quat_w, quat_x, quat_y, quat_z = pose_meta
            
            # Skip information line if it is first
            if scene_name=='Scene-ID':
                continue

            # Parse pose infomration out of string type
            traj_idx, sensor_height, view_idx,  = int(traj_idx), int(sensor_height), int(view_idx)
            x_pos, y_pos, z_pos = float(x_pos), float(y_pos), float(z_pos)
            quat_w, quat_x, quat_y, quat_z = float(quat_w), float(quat_x), float(quat_y), float(quat_z)


            rot = Quaternion((quat_w, quat_x, quat_y, quat_z))
            pose_mat = rot.to_matrix().to_4x4()
            pose_mat[0][3] = x_pos
            pose_mat[1][3] = y_pos
            pose_mat[2][3] = z_pos

            # Set camera rotation
            habitat2blender = Euler((math.pi,0,0),'XYZ').to_matrix().to_4x4()
            camera_obj.matrix_world = habitat2blender @ pose_mat

            # Update scene view layer to recalculate camera extrensic matrix
            bpy.context.view_layer.update()

            _pos, y_pos, z_pos = camera_obj.location
            quat_w, quat_x, quat_y, quat_z = camera_obj.rotation_quaternion.w, camera_obj.rotation_quaternion.x, camera_obj.rotation_quaternion.y, camera_obj.rotation_quaternion.z

            min_depth, depth_arr = render_depth()

            # Determine if rejection sampling criteria is met
            # Valid view defined as one where corner and center rays view inner mesh surface and at least 0.25m from camera origin
            corners_viewing_surface = 'Y' if camera_viewing_valid_surface(camera_obj, dist_threshold=0) else 'N'
            min_depth_passes_threshold = 'Y' if min_depth>=CONFIG['view_sampling'].getfloat('surface_distance_threshold') else 'N'
            is_valid_view = 'Y' if corners_viewing_surface and min_depth_passes_threshold else 'N'
                
            depth_arr = np.flipud(np.round(depth_arr*1000).astype(np.uint16))
            cv2.imwrite(os.path.join(SCENE_OUT_DIR, f'{SCENE_NAME}.{traj_idx:010}.{sensor_height:010}.{view_idx:010}.DEPTH.png'), depth_arr)

            accepted_view_file.write(f'{SCENE_NAME},{traj_idx:010},{sensor_height:010},{view_idx:010},{x_pos},{y_pos},{z_pos},{quat_w},{quat_x},{quat_y},{quat_z},{corners_viewing_surface},{min_depth_passes_threshold},{is_valid_view}\n')
            accepted_view_file.flush()

            view_count += 1

            if view_count%100==0:
                print(f'{view_count}/{num_views} view samples finished simulating')
    

    accepted_view_file.close()

    if verbose:
        print("***********************")
        print(f"DONE SIMULATING {view_count} TRAJECTORY SAMPLES")
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
                    usage='blender --python <path to sample_views.py> -- [options]',
                    description='Blender python script for using rejection sampling to uniformly sample valid views from the Matterport 3D semantic dataset',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-config', '--config-file', help='path to ini file containing rendering and sampling configuration', type=str)
    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-habitat', '--use-habitat-poses', help='bool specifying whether trajectory poses have been created from habitat-sim and should be used for view generation', action='store_true')
    parser.set_defaults(use_habitat_poses=False)
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
        
        scene_has_semantic_mesh = any([fl.endswith('.semantic.glb') for fl in scene_files])
        scene_has_semantic_txt = any([fl.endswith('.semantic.txt') for fl in scene_files])

        if scene_has_semantic_mesh and scene_has_semantic_txt:
            scene_has_habitat_poses = any([fl.endswith('.habitat_trajectory_poses.csv') for fl in os.listdir(os.path.join(args.output_dir, scene_dir))])
            if args.use_habitat_poses and scene_has_habitat_poses:
                sample_scene_trajectories(scene_dir_path, args.output_dir, config, verbose=args.verbose)
            else:    
                sample_scene_views(scene_dir_path, args.output_dir, config, verbose=args.verbose)
            
    if args.verbose:
        print()
        print("***********************")
        print(f"DONE SAMPLING ALL SCENES")
        print("***********************")
        print()

    bpy.ops.wm.quit_blender()