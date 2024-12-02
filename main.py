import os
import shutil
import cv2
import argparse
import re
import csv

from ssd import detect_ssd

def load_segment_masks(path: str) -> dict:
    segment_masks = {}
    for seg in ('a', 'b', 'c', 'd', 'e', 'f', 'g'):
        img = cv2.imread(f'{path}/seg-{seg}.png', cv2.IMREAD_GRAYSCALE)
        
        _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  # Threshold to binary
        segment_masks[seg] = mask
    
    if len(segment_masks) != 7:
        raise LookupError(f'Failed to load segment masks: {path}')

    return segment_masks


def process(context: dict) -> dict:
    input_image = context['image']

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    context['gray_image'] = gray_image

    results = detect_ssd(context)
    
    if context['debug']:
        output_path = context['debug_output_path']
        section = context.get('section', 'full')
        section_output_path = os.path.join(output_path, section)
        os.makedirs(section_output_path, exist_ok=True)

        if 'diag_image' in context.keys():
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

        
        context = {
            'image': image,
            'diag_image': image.copy(),
            'output_path': output_path,
            'debug_output_path': os.path.join(output_path, 'debug'),
            'section': file_name,
            'segment_masks': masks,
            'debug': debug,
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
    count = args.count
    num_frames = count

    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")
    
    cur_sec = skip

    outFile = os.path.join(output_path, 'results.csv')
    resf = open(outFile, 'w')
    res_writer = csv.writer(resf, delimiter=',')
    res_writer.writerow(['frame','temperature','profile','power','fan','time','mode'])

    trnf = any
    trn_writer = any
    debug_path = os.path.join(output_path, 'debug')

    if args.training:
        training_path = os.path.join(output_path, 'training')
        training_gt_path = os.path.join(training_path, 'gt.txt')

        os.makedirs(training_path, exist_ok=True)

        trnf = open(training_gt_path, 'w')
        trn_writer = csv.writer(trnf, delimiter='\t')

    while True:
        video.set(cv2.CAP_PROP_POS_MSEC, cur_sec * 1000)
        ret, frame = video.read()

        if not ret:
            break

        line = process_image(frame, rotate, f"frame_{cur_sec}", output_path, masks, debug)

        if line and 'results' in line.keys():
            res = line['results']

            if 'TEMP' in res.keys() \
                and 'PROFILE' in res.keys() \
                and 'POWER' in res.keys() \
                and 'FAN' in res.keys() \
                and 'TIME' in res.keys():
                res_writer.writerow([re.sub("[^0-9]", "", line['filename']), 
                    res['TEMP'],
                    res['PROFILE'],
                    res['POWER'],
                    res['FAN'],
                    res['TIME'],
                    res['MODE']])
                
                if args.training:
                    # frame_name = re.sub("[^0-9]", "", line['filename'])
                    frame_name = line['filename']
                    frame_path = os.path.join(debug_path, frame_name)
                    steps_path = os.path.join(frame_path, 'steps')

                    for idx, label in enumerate(['TEMP', 'PROFILE', 'POWER', 'FAN', 'TIME']):
                        step_file = os.path.join(steps_path, f'3-monitor-{label}-orig.png')
                        dest_filename = f'{frame_name}-{label}.png'
                        dest_file = os.path.join(training_path, dest_filename)
                        shutil.copy(step_file, dest_file)
                    
                        val = res[label]
                        if label == 'TIME' and val.strip( )== ':':
                            val = "- - - -"

                        trn_writer.writerow([dest_filename, val])
            


        if count > 0:
            num_frames = num_frames - 1
            if num_frames == 0:
                break

        cur_sec += interval

    if trnf:
        trnf.close()

    if resf:
        resf.close()

    video.release()


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    masks_path = args.masks_path

    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return

    if not os.path.exists(masks_path):
        print(f"Masks path does not exist: {masks_path}")
        return
        
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isfile(input_path):
        print(f"input file not found: {input_path}")
        return

    masks = load_segment_masks(masks_path)

    process_video(args, masks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from input path and save to output path.")
    parser.add_argument('input_path', type=str, help="Path to the input images directory or video file.")
    parser.add_argument('output_path', type=str, help="Path to the output (and debug) directory.")
    parser.add_argument('--masks_path', type=str, default='./masks', required=False, help="Path to the sevent segment display mask.")
    parser.add_argument('--skip', type=int, default=0, required=False, help="Skip number of seconds.")
    parser.add_argument('--count', type=int, default=0, required=False, help="Number of frames to process.")
    parser.add_argument('--interval', type=int, default=30, required=False, help="Processing Interval.")
    parser.add_argument('--rotate', type=str, default='auto', required=False, help="Rotation (auto|<degree>).")
    parser.add_argument('--debug', type=bool, default=False, required=False, help="Write debug image")
    parser.add_argument('--training', type=bool, default=False, required=False, help="Write debug & tensorflow training data")
    
    args = parser.parse_args()

    args.debug = args.debug or args.training

    main(args)
