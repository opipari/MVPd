import os
import sys
import csv
import shutil
import argparse
import configparser

import math
import numpy as np
import itertools, functools, operator





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

    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args(argv)


    if args.verbose:
        print()
        print(args)
        print()

    scene_directories = sorted([path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))])
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)
        scene_out_path = os.path.join(args.output_dir, scene_dir)


        src_semantic_path = os.path.join(scene_dir_path, scene_dir.split('-')[1]+'.semantic.txt')
        dst_semantic_path = os.path.join(scene_out_path, scene_dir+'.semantic.txt')
        if os.path.isfile(src_semantic_path) and os.path.isdir(scene_out_path):
            shutil.copyfile(src_semantic_path, dst_semantic_path)
            print("Copying", src_semantic_path, "to", dst_semantic_path)
            
            
