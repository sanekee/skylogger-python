
from typing import Callable
import cv2
from context import FrameContext
from debug import _debug
from utils import area


class Segment:
    def __init__(self, 
                 name: str, 
                 extract: Callable[[cv2.Mat], cv2.Mat], 
                 filter: Callable[[cv2.Mat, list[list]], list[list]]):
        self.name = name
        self.extract = extract
        self.filter = filter

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
    def __image_extractor(cls, left: float, right: float, top: float, bottom: float) -> Callable[[cv2.Mat], cv2.Mat]:
        return lambda image: image[int(top * image.shape[0]):int(bottom * image.shape[0]), 
                                   int(left * image.shape[1]):int(right * image.shape[1])]
    
    @classmethod
    def __horizontal_filter(cls, image: cv2.Mat, boxes: list[list]) -> cv2.Mat:
        return [box for box in boxes if box[2] >= 0.5 * image.shape[1] or \
                area(box[2], box[3]) >= 0.7 * area(image.shape[1], image.shape[0])]

    @classmethod
    def __vertical_filter(cls, image: cv2.Mat, boxes: list[list]) -> cv2.Mat:
        return [box for box in boxes if box[3] >= 0.5 * image.shape[0] or \
                                         area(box[2], box[3]) >= 0.5 * area(image.shape[1], image.shape[0])]

    @classmethod
    def __init_zones(cls):
        cls.__zones = {
            "top": Segment("top", 
                        cls.__image_extractor(0.1, 0.9, 0, 0.25),
                        cls.__horizontal_filter),
            "top-right": Segment("top-right", 
                        cls.__image_extractor(0.75, 1, 0.1, 0.45),
                        cls.__vertical_filter),
            "bottom-right": Segment("bottom-right",
                        cls.__image_extractor(0.65, 0.95, 0.5, 0.85),
                        cls.__vertical_filter),
            "bottom": Segment("bottom", 
                        cls.__image_extractor(0.1, 0.9, 0.75, 1),
                        cls.__horizontal_filter),
            "bottom-left": Segment("bottom-left",
                        cls.__image_extractor(0.0, 0.25, 0.5, 0.85),
                        cls.__vertical_filter),
            "top-left": Segment("top-left", 
                        cls.__image_extractor(0.0, 0.25, 0.1, 0.45),
                        cls.__vertical_filter),
            "middle": Segment("middle", 
                        cls.__image_extractor(0.1, 0.9, 0.4, 0.6),
                        cls.__horizontal_filter),
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
            zone_image = zone.extract(processed_image)

            contours, _ = cv2.findContours(zone_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours]

            orig_boxes = boxes
            boxes = zone.filter(zone_image, boxes)
            segments += '1' if len(boxes) > 0 else '0'

            def __debug_zone():
                box_image = zone.extract(image.copy())
                [cv2.rectangle(box_image, box, (0, 255, 0), 1) for box in boxes]
                if len(boxes) == 0:
                    [cv2.rectangle(box_image, box, (0, 0, 255), 2) for box in orig_boxes]
                ctx._write_step(f'{name}-{idx}-{zone.name}', box_image)

            _debug(ctx, lambda: __debug_zone())

            i+=1
        
        _debug(ctx, lambda: print(f'{ctx.name}-{name}-{idx} pattern {segments}'))

        if segments in cls.__patterns:
            return cls.__patterns[segments]

