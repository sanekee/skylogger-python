
from typing import Callable
import cv2
from context import FrameContext


class SSDResults:
    def __init__(self, name: str):
        self.name = name

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
            # cls.__instance = cls()
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
        }
    
    @classmethod
    def __init_zones(cls):
        horizontal_filter = lambda image, boxes: [box for box in boxes if (box[2] / box[3] > 1.5 and box[2] >= (0.5 * image.shape[1])) ]
        vertical_filter = lambda image, boxes: [box for box in boxes if (box[3] / box[2] > 1.5 and box[3] >= (0.5 * image.shape[0])) ]
        cls.__zones = {
            "top": Segment("top", 
                        lambda image: image[:int(0.3 * image.shape[0]), int(0.1 * image.shape[1]):int(0.9 * image.shape[1])],
                        horizontal_filter),
            "top-right": Segment("top-right", 
                        lambda image: image[int(0.1 * image.shape[0]):int(0.5 * image.shape[0]), int(0.7 * image.shape[1]):],
                        vertical_filter),
            "bottom-right": Segment("bottom-right",
                        lambda image: image[int(0.5 * image.shape[0]):int(0.9 * image.shape[0]), int(0.7 * image.shape[1]):],
                        vertical_filter),
            "bottom": Segment("bottom", 
                        lambda image: image[int(0.7 * image.shape[0]):, int(0.1 * image.shape[1]):int(0.9 * image.shape[1])],
                        horizontal_filter),
            "bottom-left": Segment("bottom-left",
                        lambda image: image[int(0.5 * image.shape[0]):int(0.9 * image.shape[0]), :int(0.3 * image.shape[1])],
                        vertical_filter),
            "top-left": Segment("top-left", 
                        lambda image: image[int(0.1 * image.shape[0]):int(0.5 * image.shape[0]), :int(0.3 * image.shape[1])],
                        vertical_filter),
            "middle": Segment("middle", 
                        lambda image: image[int(0.4 * image.shape[0]):int(0.6 * image.shape[0]), int(0.1 * image.shape[1]):int(0.9 * image.shape[1])],
                        horizontal_filter),
        }

    @staticmethod
    def __preprocess_image(image: cv2.Mat) -> cv2.Mat:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, threshold_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY) 

        return threshold_image

    @classmethod
    def detect(cls, ctx: FrameContext, name:str, idx: int, image: cv2.Mat) -> str:
        processed_image = SSD.__preprocess_image(image)

        if ctx.options.debug:
            ctx._write_step(name, f'3-digit-{idx}.png', processed_image)

            segments = ''
            i = 0
            for zone in cls.__zones.values():
                zone_image = zone.extract(processed_image)
                contours, _ = cv2.findContours(zone_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = [cv2.boundingRect(c) for c in contours]

                boxes = zone.filter(zone_image, boxes)
                box_image = zone.extract(image.copy())
                [cv2.rectangle(box_image, box, (0, 255, 0), 1) for box in boxes]
                ctx._write_step(name, f'4-digit-{idx}-zone-{i}.png', zone_image)
                ctx._write_step(name, f'4-digit-{idx}-box-{i}.png', box_image)
                i+=1
                segments += '1' if len(boxes) > 0 else '0'
        
        if segments in cls.__patterns:
            return cls.__patterns[segments]

