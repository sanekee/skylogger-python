import os
import shutil
import cv2
import argparse
import re
import numpy as np

from ssd import detect_ssd

def load_segment_masks(path: str) -> dict:
    segment_masks = {}
    for seg in ('a', 'b', 'c', 'd', 'e', 'f', 'g'):
        # Load the image of the digit (assumes the images are named num0.png to num9.png)
        img = cv2.imread(f'{path}/seg-{seg}.png', cv2.IMREAD_GRAYSCALE)
        
        # Step 2: Create a binary mask (white digit on black background)
        _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  # Threshold to binary
        segment_masks[seg] = mask
    
    return segment_masks


def process(context: dict) -> dict:
    input_image = context['image']
    output_path = context['output_path']
    section = context.get('section', 'full')
    
    section_output_path = os.path.join(output_path, section)
    os.makedirs(section_output_path, exist_ok=True)

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(os.path.join(section_output_path, 'gray.png'), gray_image)

    context['gray_image'] = gray_image

    results = detect_ssd(context)
    
    cv2.imwrite(os.path.join(section_output_path, 'diag.png'), context['diag_image'])

    if 'threshold' in context.keys():
        cv2.imwrite(os.path.join(section_output_path, 'threshold.png'), context['threshold'])

    return results

def process_image(img: object, rotate: str, file_name: str, output_path: str, masks: dict, debug: bool) -> dict:
    rotateDegree = [0]

    if rotate:
        if rotate.isdigit():
            rotateDegree = [int(rotate)]
        elif rotate == 'auto':
            rotateDegree = [0, 90, 180, 270]
        
    for degree in rotateDegree:
        image = img
        if degree == 90:
            image = cv2.transpose(image, degree)
            image = cv2.flip(image, 1)
        elif degree == 180:
            image = cv2.flip(image, -1)
        elif degree == 270:
            image = cv2.transpose(image, degree)
            image = cv2.flip(image, 0)

        
        height, width = image.shape[:2]
        if width < height:
            continue

        context = {
            'image': image,
            'diag_image': image.copy(),
            'output_path': output_path,
            'section': file_name,
            'segment_masks': masks,
            'debug': debug
        }

        results = process(context)

        if results is not None:
            return {
                'filename': file_name,
                'results': results,
                'rotate': degree,
            }

    return None


def process_video(args: object, masks: dict) -> dict:
    input_path = args.input_path
    output_path = args.output_path
    rotate = args.rotate
    debug = args.debug
    interval = args.interval 
    skip = args.skip

    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_number = 0

    csv = []
    while True:
        ret, frame = video.read()

        if not ret:  # End of video
            break

        # Save frame if it matches the interval
        if frame_number % frame_interval == 0:
            rounded_time = round(frame_number / fps)
            res = process_image(frame, rotate, f"frame_{rounded_time}", output_path, masks, debug)

            csv.append(res)

        frame_number += 1

    video.release()
    return csv


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    masks_path = args.masks_path
    rotate = args.rotate or 'auto'
    debug = args.debug or False

    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return

    if not os.path.exists(masks_path):
        print(f"Masks path does not exist: {masks_path}")
        return
        
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    csv = []
    if not os.path.isfile(input_path):
        print(f"input file not found: {input_path}")
        return

    masks = load_segment_masks(masks_path)

    csv = process_video(args, masks)

    if not csv:
        return

    outFile = os.path.join(output_path, 'results.csv')
    with open(outFile, 'w') as f:
        f.write('frame,temperature,profile,power,fan,time,mode\n')
        for line in csv:
            f.write(re.sub("[^0-9]", "", line['filename']) + ',')
            if line['results']:
                res = line['results']
                f.write(f"{res['TEMP']},{res['PROFILE']},{res['POWER']},{res['FAN']},{res['TIME']},{res['MODE']}")
            f.write('\n')

        f.write('\n')
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from input path and save to output path.")
    parser.add_argument('input_path', type=str, help="Path to the input images directory or video file.")
    parser.add_argument('output_path', type=str, help="Path to the output (and debug) directory.")
    parser.add_argument('--masks_path', type=str, default='./masks', required=False, help="Path to the sevent segment display mask.")
    parser.add_argument('--skip', type=int, default=0, required=False, help="Skip number of seconds.")
    parser.add_argument('--interval', type=int, default=30, required=False, help="Processing Interval.")
    parser.add_argument('--rotate', type=str, default='auto', required=False, help="Rotation (auto|<degree>).")
    parser.add_argument('--debug', type=bool, default=False, required=False, help="Write debug image")
    
    args = parser.parse_args()

    main(args)
