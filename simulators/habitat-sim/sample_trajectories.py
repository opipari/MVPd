import os
import sys
import argparse
import configparser

import math
import numpy as np
import magnum as mn
import itertools, functools, operator

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

import cv2
from PIL import Image 
from scipy.spatial.transform import Rotation as R



def make_cfg(scene, scene_config, CONFIG):
    """Generate a configuration object for instantiating habitat-sim simulator.

    Keyword arguments:
    scene -- string path to habitat-sim compatible scene file (e.g. .glb formatted mesh)
    scene_config -- string path to habitat-sim compatible scene configuration file in .json format
    CONFIG -- configparser object holding config settings for sensor specification in habitat-sim
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene
    sim_cfg.scene_dataset_config_file = scene_config
    sim_cfg.enable_physics = False

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [CONFIG['sensor_spec'].getfloat('image_height'), CONFIG['sensor_spec'].getfloat('image_width')]
    color_sensor_spec.position = [0.0, CONFIG['sensor_spec'].getint('sensor_height')/1000.0, 0.0]
    color_sensor_spec.hfov = mn.Deg(CONFIG['sensor_spec'].getfloat('hfov'))
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    # depth_sensor_spec = habitat_sim.CameraSensorSpec()
    # depth_sensor_spec.uuid = "depth_sensor"
    # depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    # depth_sensor_spec.resolution = [CONFIG['sensor_spec'].getfloat('image_height'), CONFIG['sensor_spec'].getfloat('image_width')]
    # depth_sensor_spec.position = [0.0, CONFIG['sensor_spec'].getint('sensor_height')/1000.0, 0.0]
    # depth_sensor_spec.hfov = mn.Deg(CONFIG['sensor_spec'].getfloat('hfov'))
    # depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    # sensor_specs.append(depth_sensor_spec)

    # semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    # semantic_sensor_spec.uuid = "semantic_sensor"
    # semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    # semantic_sensor_spec.resolution = [CONFIG['sensor_spec'].getfloat('image_height'), CONFIG['sensor_spec'].getfloat('image_width')]
    # semantic_sensor_spec.position = [0.0, CONFIG['sensor_spec'].getint('sensor_height')/1000.0, 0.0]
    # semantic_sensor_spec.hfov = mn.Deg(CONFIG['sensor_spec'].getfloat('hfov'))
    # semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    # sensor_specs.append(semantic_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_rotation(point, tangent):
    """Helper function to define a quaternion rotation towards tangent direction from point. Assumes world-up direction of +Y

    Keyword arguments:
    point -- numpy array of length 3 defining 3D point in euclidean space
    tangent -- numpy array of length 3 defining viewing direction in euclidean space
    """
    tangent_orientation_matrix = mn.Matrix4.look_at(
        point, point + tangent, np.array([0, 1.0, 0])
    )
    tangent_orientation_q = mn.Quaternion.from_matrix(
        tangent_orientation_matrix.rotation()
    )
    rotation = utils.quat_from_magnum(tangent_orientation_q)
    
    return rotation


def smooth_path(path_points, forward_step=0.05, turn_step=0.5):
    """Helper function to convert sparse waypoints from habitat-sim navmesh into smooth trajectory to model an agent moving in scene.
        Motion is linear along each segment (point-to-point) then linear along axis-angle rotation from tangent-to-tangent.

    Keyword arguments:
    path_points -- list of numpy arrays. Each array defining navmesh waypoint in 3D space.
    forward_step -- float describing maximum step size for linear displacement from point-to-point
    turn_step -- float describing maximum step size for axis-angle rotation in degrees
    """

    # Define starting point and starting viewing direction
    current_position = path_points[0]
    tangent = path_points[1] - current_position
    current_rotation = get_rotation(current_position, tangent)

    # Add starting point and viewing direction to smoothed path list
    expanded_targets = [(current_position, current_rotation)]

    # Iterate over each waypoint from navmesh. For each point, expand intermediate poses for smoothed trajectory.
    for ix, point in enumerate(path_points):

        # While not reached the final waypoint, apply linear displacement steps to move towards 'next' waypoint
        if ix < len(path_points) - 1:

            # While loop to continually move towards 'next' waypoint
            reached_target = False
            while not reached_target:
                tangent = path_points[ix + 1] - current_position
                distance_to_target = np.linalg.norm(tangent)
                
                # If agent within a maximum forward step, consider waypoint reached, update state correspondingly
                if distance_to_target < forward_step:
                    reached_target = True
                    current_position = path_points[ix + 1]
                    # Add expanded waypoint to smoothed path list and move to next waypoint
                    expanded_targets.append((current_position, current_rotation))
                    break

                # Otherwise, take maximum step in straight-line direction of waypoint
                direction_to_target = tangent / distance_to_target
                current_position = current_position + (forward_step * direction_to_target)
                # Add expanded waypoint to smoothed path list and continue linear movement
                expanded_targets.append((current_position, current_rotation))

            
        # After reaching 'next' waypoint in position, agent must rotate towards 'next-next' waypoint
        # This rotation will ensure future linear steps align with camera viewing direction
        if ix < len(path_points) - 2:

            # While loop to continually rotate towards next tangent viewing direction
            reached_target = False
            while not reached_target:
                next_tangent = path_points[ix + 2] - path_points[ix + 1]
                next_rotation = get_rotation(current_position, next_tangent)

                # Use axis-angle representation to calculate minimum rotation needed to align agent view direction with tangent
                rotation_diff = next_rotation * current_rotation.conjugate()
                theta, w = utils.quat_to_angle_axis(rotation_diff)
                
                # Handle case where axis-angle representation does not align with notion of shortest path
                # Essentially, don't allow rotations outside the range of (-180, +180)
                # This range allows all rotations and ensures shortest time spent rotating to next view direction
                theta_deg = math.degrees(theta)%360
                if theta_deg > 180:
                    theta_deg -= 360
                theta = math.radians(theta_deg)

                # If agent within maximum step of desired view direction, consider view direction reached, update state correspondingly
                if abs(theta) < math.radians(turn_step):
                    reached_target = True
                    current_rotation = next_rotation
                    # Add expanded waypoint to smoothed path list and move to next waypoint
                    expanded_targets.append((current_position, current_rotation))
                    break
                
                # Otherwise, take maximum step-sized rotation along correct axis to move view direction towards tangent
                theta = np.sign(theta) * math.radians(turn_step)
                current_rotation = utils.quat_from_angle_axis(theta, w) * current_rotation
                # Add expanded waypoint to smoothed path list and move to next waypoint and continue rotational displacement
                expanded_targets.append((current_position, current_rotation))

    # Return expanded list of waypoints for view sampling
    return expanded_targets


def sample_scene_trajectories(SCENE_DIR, SCENE_OUT_DIR, CONFIG, render_images=False, append_samples=False, verbose=True):
    
    SCENE_NAME = SCENE_DIR.split('/')[-1]
    SCENE_FILE = SCENE_NAME.split('-')[-1]+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME.split('-')[-1]+'.semantic.glb'
    SCENE_CONFIG_FILE = f"hm3d_annotated_{SCENE_DIR.split('/')[-2]}_basis.scene_dataset_config.json"
    SCENE_CONFIG_PATH = os.path.join('/'.join(SCENE_DIR.split('/')[:-2]), SCENE_CONFIG_FILE)


    if verbose:
        print()
        print("********************")
        print(f"SAMPLING TRAJECTORIES FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()

    os.makedirs(SCENE_OUT_DIR, exist_ok=True)



    if verbose:
        print()
        print("***********************")
        print("INITIALIZING SCENE")
    
    cfg = make_cfg(os.path.join(SCENE_DIR, SCENE_FILE), SCENE_CONFIG_PATH, CONFIG)
    sim = habitat_sim.Simulator(cfg)

    if not sim.pathfinder.is_loaded:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        # assert navmesh_success
    

    agent = sim.initialize_agent(0)


    if verbose:
        print("DONE INITIALIZING SCENE")
        print("***********************")
        print()

    
    csv_pose_path = os.path.join(SCENE_OUT_DIR, f"{SCENE_NAME}.habitat_trajectory_poses.csv")
    if append_samples and os.path.isfile(csv_pose_path):
        accepted_view_file = open(csv_pose_path, "a")
    else:
        accepted_view_file = open(csv_pose_path, "w")
        accepted_view_file.write('Scene-ID,Trajectory-ID,Sensor-Height,View-ID,X-Position,Y-Position,Z-Position,W-Quaternion,X-Quaternion,Y-Quaternion,Z-Quaternion\n')
        accepted_view_file.flush()
    


    if verbose:
        print()
        print("***********************")
        print(f"INITIATING RENDERING")
    


    valid_path_count = 0 
    valid_view_count = 0

    # Seed pathfinder for pseudo random sampling
    seed = 0
    sim.pathfinder.seed(seed)

    # Define lambda function and list of sampled navmesh points to avoid duplicate trajectory sampling
    sampled_start_goal_points = []
    points_in_arr_list = lambda start_arr, goal_arr, list_of_start_goal_tuples : any([(start_arr==start).all() and (goal_arr==goal).all() for (start, goal) in list_of_start_goal_tuples])
    
    # Simulate at least minimum number desired trajectories and image frames for this specific scene
    while valid_path_count < CONFIG['trajectory_settings'].getint('minimum_trajectories_per_scene') or valid_view_count < CONFIG['trajectory_settings'].getint('minimum_frames_per_scene'):

        # Sample valid points on the NavMesh for agent spawn location and pathfinding goal
        sample_start = sim.pathfinder.get_random_navigable_point()
        sample_goal = sim.pathfinder.get_random_navigable_point()

        # Don't create duplicate trajectories
        if points_in_arr_list(sample_start, sample_goal, sampled_start_goal_points):
            continue

        # Record the start and goal points to avoid future duplicate trajectories
        sampled_start_goal_points.append((sample_start, sample_goal))

        # Use ShortestPath module to compute path between samples
        path = habitat_sim.ShortestPath()
        path.requested_start = sample_start
        path.requested_end = sample_goal

        found_path = sim.pathfinder.find_path(path)        
        if found_path:

            # Post process points to form a smooth trajectory
            path_points = smooth_path(path.points, 
                                        forward_step=CONFIG['trajectory_settings'].getfloat('forward_step'), 
                                        turn_step=CONFIG['trajectory_settings'].getfloat('turn_step'))

            # Only accept the trajectory if it has at least a minimum number of frames
            if len(path_points) > CONFIG['trajectory_settings'].getint('minimum_frames_per_trajectory') and len(path_points) <= CONFIG['trajectory_settings'].getint('maximum_frames_per_trajectory'):
                
                trajectory_view_count = 0
                agent_state = habitat_sim.AgentState()
                
                # Iterate over each pose in the post-processed trajectory
                for point, rotation in path_points:
                        
                        # Set agent to specified pose
                        agent_state.position = point
                        agent_state.rotation = rotation
                        agent.set_state(agent_state)

                        if render_images:
                            observations = sim.get_sensor_observations()
                            # rgb, depth, semantic = observations["color_sensor"], observations["depth_sensor"], observations["semantic_sensor"]

                            rgb_img = Image.fromarray(observations["color_sensor"], mode="RGBA")
                            rgb_img.save(os.path.join(SCENE_OUT_DIR, f'{SCENE_NAME}.{valid_path_count:010}.{CONFIG["sensor_spec"].getint("sensor_height"):010}.{trajectory_view_count:010}.RGB.{0:010}.png'))


                        # Don't use agent state for pose since sensor may be translated from the local origin of the agent (e.g. to simulate agent height)
                        # x, y, z = agent_state.position
                        # quat_w, quat_x, quat_y, quat_z = [agent_state.rotation.real]+list(agent_state.rotation.imag)

                        # Extract pose of camera sensor
                        sensor_pose = agent.get_state().sensor_states['color_sensor']
                        x, y, z = sensor_pose.position
                        quat_w, quat_x, quat_y, quat_z = [sensor_pose.rotation.real]+list(sensor_pose.rotation.imag)

                        # Write pose information to file
                        accepted_view_file.write(f'{SCENE_NAME},{valid_path_count:010},{CONFIG["sensor_spec"].getint("sensor_height"):010},{trajectory_view_count:010},{x},{y},{z},{quat_w},{quat_x},{quat_y},{quat_z}\n')
                        accepted_view_file.flush()

                        trajectory_view_count += 1
                        valid_view_count += 1
                valid_path_count += 1

    sim.close()
    accepted_view_file.close()

    if verbose:
        print("***********************")
        print(f"DONE SIMULATING {valid_path_count} TRAJECTORY SAMPLES")
        print(f"ACCEPTED {valid_view_count} VALID VIEWS")



if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='sample_trajectories',
                    usage='python <path to sample_trajectories.py> -- [options]',
                    description='Python script for sampling trajectories from the Matterport 3D semantic dataset using habitat sim',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/habitat-sim')

    parser.add_argument('-config', '--config-file', help='path to ini file containing rendering and sampling configuration', type=str)
    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-append', '--append-samples', help='bool specifying whether new samples should append or overwrite existing trajectories (default False)', action='store_true')
    parser.set_defaults(append_samples=False)
    parser.add_argument('-render', '--render-images', help='bool specifying whether to render RGB images with each view (default False)', action='store_true')
    parser.set_defaults(render_images=False)
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

        scene_has_semantic_mesh = any([fl.endswith('.semantic.glb') for fl in scene_files])
        scene_has_semantic_txt = any([fl.endswith('.semantic.txt') for fl in scene_files])

        if scene_has_semantic_mesh and scene_has_semantic_txt:
            sample_scene_trajectories(scene_dir_path, scene_out_path, config, render_images=args.render_images, append_samples=args.append_samples, verbose=args.verbose)

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE RENDERING ALL SCENES")
        print("***********************")
        print()