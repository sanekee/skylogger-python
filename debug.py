
from cmath import rect
import math
from multiprocessing import context
from cv2 import Mat
import cv2
from aoi import Rect
from context import FrameContext
from utils import calculate_angle, find_central_box_index, midpoint

def _debug_projection(ctx: FrameContext, rects: list[Rect]):
    img = ctx.image.copy()
    color: cv2.typing.Scalar = (255, 255, 255)
    color2: cv2.typing.Scalar = (0, 255, 255)

    cidx = find_central_box_index(rects)
    rect1 = rects[cidx]
    projected_rect1 = rect1.projected()
    center1 = projected_rect1.center()
    for rect2 in rects:
        cv2.rectangle(img, rect2.to_list(), color, 2) 
        projected_rect = rect2.projected()
        cv2.rectangle(img, projected_rect.to_list(), color2, 1) 

        if rect2 == rect1:
            continue

        center2 = projected_rect.center()
        line_length = int(math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2))
        ratio = round(line_length / rect1.h, 2)
        angle = round(calculate_angle(center1, center2), 2)

        text = f"A:{angle}, L:{line_length}, R:{ratio}"

        mid_pt = midpoint(center1, center2)

        cv2.line(img, center1, center2, color2, 1)
        cv2.putText(img, text, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 1)

    ctx._write_step("projection", img)


def _debug_displays(ctx: FrameContext, rects: dict[str, Rect]):
    color: cv2.typing.Scalar = (255, 255, 188)

    rect = rects["POWER"]
    _,_,_,h1 = rect.to_list()
    center1 = rect.projected().center()

    for name, rect2 in rects.items():
        _write_box(ctx, rect2, name, color)

        if name == "POWER":
            continue

        center2 = rect2.projected().center()
        line_length = int(math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2))
        ratio = round(line_length / h1, 2)
        angle = round(calculate_angle(center1, center2), 2)

        text = f"A:{angle}, L:{line_length}, R:{ratio}"

        mid_pt = midpoint(center1, center2)

        cv2.line(ctx._get_debug_image(), center1, center2, color, 2)
        cv2.putText(ctx._get_debug_image(), text, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def _debug_boxes(ctx: FrameContext, boxes: list[list]):
    color: cv2.typing.Scalar = (0, 255, 255)

    for i, box2 in enumerate(boxes):
        _write_box(ctx, box2, '', color)

def _write_box(ctx: FrameContext, rect: Rect, name:str, color: cv2.typing.Scalar): 
    cv2.rectangle(ctx._get_debug_image(), rect.to_list(), color, 1)

    if name != '':
        text = f'{name}, A:{rect.area()}, P:[{rect.x}, {rect.y}], D:[{rect.w}x{rect.h}]'
        cv2.putText(ctx._get_debug_image(), text, [rect.x, rect.y - 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
