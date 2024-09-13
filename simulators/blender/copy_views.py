import os
import shutil
import sys
import argparse



if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='copy_views',
                    usage='python <path to copy_views.py> -- [options]',
                    description='Python script for copying view samples between machines',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-src', '--source-dataset-dir', help='path to directory where existing dataset with sampled views exists', type=str)
    parser.add_argument('-dst', '--destination-dataset-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args(argv)

    if args.verbose:
        print()
        print(args)
        print()

    scene_directories = [path for path in os.listdir(args.source_dataset_dir) if os.path.isdir(os.path.join(args.source_dataset_dir, path))]
    for scene_dir in scene_directories:
        scene_src_path = os.path.join(args.source_dataset_dir, scene_dir)
        scene_dst_path = os.path.join(args.destination_dataset_dir, scene_dir)        

        os.makedirs(scene_dst_path, exist_ok=True)

        for fl in os.listdir(scene_src_path):
            if fl.endswith('.csv'):
                shutil.copyfile(os.path.join(scene_src_path, fl), os.path.join(scene_dst_path, fl))
        

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE COPYING ALL SCENES")
        print("***********************")
        print()