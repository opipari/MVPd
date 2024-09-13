import math
import os
import random
import sys
import json

import imageio
import magnum as mn
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut


def save_sample(index, scene_name, dest_dir, rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    rgb_img = rgb_img.convert('RGB')
    rgb_img.save(os.path.join(dest_dir, f'scene.{scene_name}.frame.{index:04}.color.jpg'))

    if semantic_obs.size != 0:
        np.save(os.path.join(dest_dir, f'scene.{scene_name}.frame.{index:04}.semantic.npy'), semantic_obs)
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        semantic_img.save(os.path.join(dest_dir, f'scene.{scene_name}.frame.{index:04}.semantic.png'))


    if depth_obs.size != 0:
        np.save(os.path.join(dest_dir, f'scene.{scene_name}.frame.{index:04}.depth.npy'), depth_obs)
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        depth_img.save(os.path.join(dest_dir, f'scene.{scene_name}.frame.{index:04}.depth.png'))



def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def save_map(dest_dir, topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.savefig(os.path.join(dest_dir,"map.png"))




def do_sim(sim_ix, sim_settings, scene_name, num_paths=1):
    cfg = make_cfg(sim_settings)
    # Needed to handle out of order cell run in Colab
    try:  # Got to make initialization idiot proof
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)


    for path_ix in range(num_paths):
        # the randomness is needed when choosing the actions
        random.seed(sim_settings["seed"]+path_ix)
        sim.seed(sim_settings["seed"]+path_ix)

        # Set agent state
        agent = sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
        agent.set_state(agent_state)

        # Get agent state
        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)



        # With a valid PathFinder instance:
        if not sim.pathfinder.is_loaded:
            print("Pathfinder not initialized, aborting.")
        else:
            seed = 4+path_ix  # @param {type:"integer"}
            sim.pathfinder.seed(seed)

            # Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
            sample1 = sim.pathfinder.get_random_navigable_point()
            sample2 = sim.pathfinder.get_random_navigable_point()

            # Use ShortestPath module to compute path between samples.
            path = habitat_sim.ShortestPath()
            path.requested_start = sample1
            path.requested_end = sample2
            found_path = sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            path_points = path.points
            # Success, geodesic path length, and 3D points can be queried.
            print("found_path : " + str(found_path))
            print("geodesic_distance : " + str(geodesic_distance))
            print("path_points : " + str(path_points))

            # Display trajectory (if found) on a topdown map of ground floor
            if found_path:
                # Place agent and render images at trajectory points (if found).
                print("Rendering observations at path points:")
                tangent = path_points[1] - path_points[0]
                agent_state = habitat_sim.AgentState()
                for ix, point in enumerate(path_points):
                    if ix < len(path_points) - 1:
                        tangent = path_points[ix + 1] - point
                        agent_state.position = point
                        tangent_orientation_matrix = mn.Matrix4.look_at(
                            point, point + tangent, np.array([0, 1.0, 0])
                        )
                        tangent_orientation_q = mn.Quaternion.from_matrix(
                            tangent_orientation_matrix.rotation()
                        )
                        agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                        agent.set_state(agent_state)

                        observations = sim.get_sensor_observations()
                        rgb = observations["color_sensor"]
                        semantic = observations["semantic_sensor"]
                        depth = observations["depth_sensor"]

                        save_sample(sim_ix, scene_name, dest_dir, rgb, semantic, depth)
                        sim_ix += 1
    sim.close()
    return sim_ix

#--test-scene ./datasets/Replica-Dataset/replica_v1/room_0/habitat/mesh_semantic.ply 
#--test-scene-config ./datasets/Replica-Dataset/replica_v1/replica.scene_dataset_config.json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", dest="source_dataset")
    parser.add_argument("--source-split", dest="source_split")
    parser.add_argument("--dest-dir", dest="dest_dir")
    parser.add_argument("--sim-height", dest="sim_height")
    parser.set_defaults(source_dataset="./zeroshot_rgbd/datasets/matterport/HM3D/",    
                        source_split="val",
                        dest_dir="./zeroshot_rgbd/datasets/VaryingPerspectiveDataset/data_1.65m/",
                        sim_height=1.65)
    args, _ = parser.parse_known_args()

    source_dataset = args.source_dataset
    source_split = args.source_split
    dest_dir = args.dest_dir
    sim_height = float(args.sim_height)

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)


    test_scene_dataset = os.path.join(source_dataset, f'hm3d_annotated_{source_split}_basis.scene_dataset_config.json')
    with open(test_scene_dataset, 'r') as fp:
        split_meta = json.load(fp)

    
    semantic_scenes = [path.split('/')[0] for path in split_meta["stages"]["paths"][".glb"]]
    

    sim_index = 0
    for scene in semantic_scenes:
        scene_name = scene.split("-")[1]
        scene_path = os.path.join(source_dataset, source_split, scene, f'{scene_name}.basis.glb')
    

        sim_settings = {
            "width": 800,  # Spatial resolution of the observations
            "height": 534,
            "scene": scene_path,  # Scene path
            "scene_dataset": test_scene_dataset,  # the scene dataset configuration files
            "default_agent": 0,
            "sensor_height": sim_height,  # Height of sensors in meters
            "color_sensor": True,  # RGB sensor
            "depth_sensor": True,  # Depth sensor
            "semantic_sensor": True,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
        }

        sim_index = do_sim(sim_index, sim_settings, scene_name, num_paths=2)