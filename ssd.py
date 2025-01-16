
import math
from typing import Callable, Tuple
import cv2
import numpy as np
from context import FrameContext
from debug import _debug
from utils import area


class Segment:
    def __init__(self, 
                name: str, 
                filter: Callable[[cv2.Mat, list[list]], list[list]],
                mask: Callable[[FrameContext, cv2.Mat], Tuple[cv2.Mat, list[list]]] = None):
        self.name = name
        self.filter = filter
        self.mask = mask

class SSD:
    __instance = None 
    __patterns : dict[str, str] = {}
    __zones = {}

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__init_patterns()
            cls.__init_zones()
        
        return cls.__instance

    @classmethod
    def __init_patterns(cls):
        cls.__patterns = {
            "0000000": None,
            "1111110": '0',
            "0110000": '1',
            "1101101": '2',
            "1111001": '3',
            "0110011": '4',
            "1011011": '5',
            "1011111": '6',
            "1110000": '7',
            "1111111": '8',
            "1111011": '9',
            "1110111": 'A',
            "1000110": 'T',
            "1001110": 'C',
            "0001110": 'L',
            "0000001": '-',
        }

    @classmethod
    def __horizontal_filter(cls, image: cv2.Mat, boxes: list[list]) -> cv2.Mat:
        return [box for box in boxes if box[2] >= 0.5 * image.shape[1] or \
                area(box[2], box[3]) >= 0.7 * area(image.shape[1], image.shape[0])]

    @classmethod
    def __vertical_filter(cls, image: cv2.Mat, boxes: list[list]) -> cv2.Mat:
        return [box for box in boxes if box[3] >= 0.4 * image.shape[0] or \
                                         area(box[2], box[3]) >= 0.5 * area(image.shape[1], image.shape[0])]
    
    @classmethod
    def _segment_mask(cls, points: list[list[int]]) -> Callable[[cv2.Mat], cv2.Mat]:
        def __apply_mask(ctx: FrameContext, name: str, zone_name: str, image: cv2.Mat) -> cv2.Mat:
            h, w = image.shape

            w -= 1 
            h -= 1
            
            min_x = w 
            min_y = h
            max_x = 0
            max_y = 0
            coords : list[list[int]] = []
            for p0, p1 in points:
                x = int(p0 * w)
                y = int(p1 * h)

                min_x = min(x, min_x)
                min_y = min(y, min_y)
                max_x = max(x, max_x)
                max_y = max(y, max_y)

                coords.append([x, y])
                
            mask = np.zeros_like(image, dtype=np.uint8)
            arr = np.array(coords, dtype=np.int32)
            cv2.fillPoly(mask, [arr], 255)

            masked = cv2.bitwise_and(image, image, mask=mask)
            zoned = masked[min_y:max_y, min_x:max_x]
            
            def __debug_mask():
                img = image.copy()
                cv2.polylines(img, [arr], True, (255, 255, 255), 1)
                ctx._write_step(f'{name}-{zone_name}-lines', img)
            
            _debug(ctx, lambda: __debug_mask())

            return zoned, [min_x, min_y, max_x - min_x, max_y - min_y]

        return __apply_mask


    @classmethod
    def __init_zones(cls):
        cls.__zones = {
            "top": Segment("top", 
                        cls.__horizontal_filter,
                        cls._segment_mask([[0,0], [1,0], [0.5, 0.25]])),
            "top-right": Segment("top-right", 
                        cls.__vertical_filter,
                        cls._segment_mask([[1,0], [1,0.5], [0.5, 0.25]])),
            "bottom-right": Segment("bottom-right",
                        cls.__vertical_filter,
                        cls._segment_mask([[1,0.5], [1,1], [0.5, 0.75]])),
            "bottom": Segment("bottom", 
                        cls.__horizontal_filter,
                        cls._segment_mask([[0,1], [1,1], [0.5, 0.75]])),
            "bottom-left": Segment("bottom-left",
                        cls.__vertical_filter,
                        cls._segment_mask([[0,1], [0,0.5], [0.5, 0.75]])),
            "top-left": Segment("top-left", 
                        cls.__vertical_filter,
                        cls._segment_mask([[0,0], [0,0.5], [0.5, 0.25]])),
            "middle": Segment("middle", 
                        cls.__horizontal_filter,
                        cls._segment_mask([[0,0.5], [0.5,0.25], [1, 0.5], [0.5, 0.75]])),
        }

    @staticmethod
    def __preprocess_image(image: cv2.Mat) -> cv2.Mat:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, threshold_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY) 

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
        dilated = cv2.dilate(threshold_image , kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        return closed

    @classmethod
    def detect(cls, ctx: FrameContext, name:str, idx: int, image: cv2.Mat) -> str:
        processed_image = SSD.__preprocess_image(image)

        ctx._write_step(f'{name}-{idx}', processed_image)
           
        segments = ''
        i = 0
        for zone in cls.__zones.values():
            zone_image, zone_box = zone.mask(ctx, name, f'{idx}-{zone.name}', processed_image)

            contours, _ = cv2.findContours(zone_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours]

            orig_boxes = boxes
            boxes = zone.filter(zone_image, boxes)

            segments += '1' if  len(boxes) > 0 else '0'

            def __debug_zone():
                x, y, w, h = zone_box
                box_image = image[y:y+h, x:x+w].copy()
                [cv2.rectangle(box_image, box, (0, 255, 0), 1) for box in boxes]
                if len(boxes) == 0:
                    [cv2.rectangle(box_image, box, (0, 0, 255), 2) for box in orig_boxes]
                ctx._write_step(f'{name}-{idx}-{zone.name}', box_image)

            _debug(ctx, lambda: __debug_zone())

            i+=1
        
        _debug(ctx, lambda: print(f'{ctx.name}-{name}-{idx} pattern {segments}'))

        if segments in cls.__patterns:
            return cls.__patterns[segments]

