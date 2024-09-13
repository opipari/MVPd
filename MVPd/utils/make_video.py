import os
import argparse
from pathlib import Path

from MVPd.utils.MVPdHelpers import make_video


if __name__=="__main__":

    parser = argparse.ArgumentParser(
                    prog='make_video.py',
                    usage='python <path to make_video.py> -- [options]',
                    description='Python script for converting MVPd sequence into a gif file for visualization',
                    epilog='For more information, see: https://github.com/opipari/MVPd')

    parser.add_argument('-video', '--video-dir', help='path to directory of rendered video frames', type=str)
    parser.add_argument('-output', '--output-file', help='path to directory where output video file', type=str, default=None)

    args = parser.parse_args()       

    output_file = args.output_file
    if output_file is None:
        video_parent_dir = Path(args.video_dir).parent
        video_name = str(Path(args.video_dir).stem)+'.mp4'
        output_file = os.path.join(video_parent_dir, video_name)
    print(output_file)
    make_video(args.video_dir, output_file)
