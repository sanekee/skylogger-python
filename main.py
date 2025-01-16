import os
import shutil
from typing import List, Optional
import cv2
import argparse
import re
import csv

from context import Context, FrameContext, Settings, Options
from skywalker import SkyWalker, Result

def write_result(ctx: Context, results: list[Result]):
    if len(results) == 0:
        return

    outFile = os.path.join(ctx.settings.output_path, 'results.csv')
    with open(outFile, 'w') as f:
        wrt = csv.writer(f, delimiter=',')
        wrt.writerow(['name', 'time', 'temperature','profile','power','fan','mode'])

        for res in results:
            wrt.writerow([res.name, res.time, res.temperature, res.profile, res.power, res.fan, res.mode])


def process_image(ctx: FrameContext) -> Optional[Result]:
    degrees: List[int] = [0]
    options: Options = ctx.options

    if options.rotate:
        if options.rotate.isdigit():
            degrees = [int(options.rotate)]
        elif options.rotate == 'auto':
            degrees = [0, 90, 180, 270]
        
    for degree in degrees:
        image = ctx.image
        if degree == 90:
            image = cv2.transpose(image, degree)
            image = cv2.flip(image, 1)
        elif degree == 180:
            image = cv2.flip(image, -1)
        elif degree == 270:
            image = cv2.transpose(image, degree)
            image = cv2.flip(image, 0)

        res = SkyWalker(ctx).detect()
        if res is not None:
            return res
            
    return None

def process_video(ctx: Context):
    settings: Settings = ctx.settings
    options: Options = ctx.options

    video = cv2.VideoCapture(settings.input_path)
    if not video.isOpened():
        raise ValueError(f"Cannot open video file: {settings.input_path}")
    
    cur_sec = options.skip
    num_frames = options.count

    results: list[Result] = []

    while True:
        video.set(cv2.CAP_PROP_POS_MSEC, cur_sec * 1000)
        ret, frame = video.read()

        if not ret:
            break
        
        line = process_image(ctx.new_frame_context(f"frame_{cur_sec}", frame))

        if line is not None:
            results.append(line)

        if options.count > 0:
            num_frames = num_frames - 1
            if num_frames == 0:
                break

        cur_sec += options.interval

    video.release()

    write_result(ctx, results)

def main(args):
    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return

    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isfile(input_path):
        print(f"input file not found: {input_path}")
        return

    context = Context(args)
    process_video(context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from input path and save to output path.")
    parser.add_argument('input_path', type=str, help="Path to the input images directory or video file.")
    parser.add_argument('output_path', type=str, help="Path to the output (and debug) directory.")
    parser.add_argument('--skip', type=int, default=0, required=False, help="Skip number of seconds.")
    parser.add_argument('--count', type=int, default=0, required=False, help="Number of frames to process.")
    parser.add_argument('--interval', type=int, default=30, required=False, help="Processing Interval.")
    parser.add_argument('--rotate', type=str, default='auto', required=False, help="Rotation (auto|<degree>).")
    parser.add_argument('--debug', type=bool, default=False, required=False, help="Write debug image")
    
    main(parser.parse_args())
