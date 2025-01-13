
from cmath import rect
import math
from multiprocessing import context
from cv2 import Mat
import cv2
from context import FrameContext
from utils import calculate_angle, midpoint, rect_center


def debug_displays(ctx: FrameContext, boxes: dict[str, list]):
    color: cv2.typing.Scalar = (255, 255, 188)

    box1 = boxes["POWER"]
    _,_,_,h1 = box1
    center1 = rect_center(box1)

    for name, box2 in boxes.items():
        write_box(ctx, box2, name, color)

        if name == "POWER":
            continue

        center2 = rect_center(box2) 
        line_length = int(math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2))
        ratio = round(line_length / h1, 2)
        angle = round(calculate_angle(center1, center2), 2)

        text = f"A:{angle}, L:{line_length}, R:{ratio}"

        mid_pt = midpoint(center1, center2)

        cv2.line(ctx._debug_image, center1, center2, color, 2)
        cv2.putText(ctx._debug_image, text, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    ctx._write_debug()

def write_box(ctx: FrameContext, box: list, name:str, color: cv2.typing.Scalar): 
    text = f'{name}, A:{box[2]*box[3]}, P:[{box[0], box[1]}], D:[{box[2]}x{box[3]}]'

    cv2.rectangle(ctx._debug_image, box, color, 1)
    cv2.putText(ctx._debug_image, text, [box[0], box[1] - 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
