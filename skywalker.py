
import math
from typing import List
from cv2 import Mat
import cv2
from context import FrameContext
from debug import debug_displays
from utils import calculate_angle, calculate_p2, find_box2, find_central_box_index, group, merge, rect_center


class Section:
    def __init__(self, name: str, angle: float, length: float):
        self.name = name
        self.angle = angle
        self.length = length

class Display:
    def __init__(self, name: str, rect: list):
        self.name = name
        self.rect = rect

class Result:
    def __init__(self, name: str, temperature: int, profile: int, power: int, fan: int, time: str, mode: str):
        self.name = name
        self.temperature = temperature
        self.profile = profile
        self.power = power
        self.fan = fan
        self.time = time
        self.mode = mode

        
class SkyWalker():
    def __init__(self, ctx: FrameContext):
        self.ctx = ctx

        self.__init_sections()
        self.minAreaSize = 50

    def __init_sections(self):
        self.__sections = {
            "TEMPERATURE": Section("TEMPERATURE", -149.265, 5.2),
            "PROFILE": Section("PROFILE", -50.26, 3.24),
            "POWER": Section("POWER", 0, 0),
            "FAN": Section("FAN", 0.0, 5.25),
            "TIME": Section("TIME", 165.50, 5.0),
            "MODE_PREHEAT": Section("MODE_PREHEAT", 115.49, 5.0),
            "MODE_ROAST": Section("MODE_ROAST", 88.01, 4.59),
            "MODE_COOL": Section("MODE_COOL", 59.01, 51.38),
        }

    def __preprocess_image(self) -> Mat:
        ctx = self.ctx
        gray_image = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY) 
        return threshold


    def __find_aoi(self, image: Mat) -> list:
        def filter_area(contours, min):
            for c in contours:
                if cv2.contourArea(c) > min:
                    yield c

        contours, _ = cv2.findContours( 
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        contours = filter_area(contours, 50)

        boxes = [cv2.boundingRect(c) for c in contours]

        rows = group(boxes)
        rows = merge(rows)

        return [item for row in rows for item in row]


    def __detect_displays(self, threshold_image) -> List[Display]:
        boxes = self.__find_aoi(threshold_image)

        displays: dict[str, Display]= {}

        cidx = find_central_box_index(boxes)
        box1 = boxes[cidx]

        displays['POWER'] = Display('POWER', box1)
        
        for section in self.__sections.values():
            if section.name == 'POWER':
                continue

            pt_check = calculate_p2(box1, section.length, section.angle)
            box2 = find_box2(pt_check, boxes)

            if box2 is None:
                continue

            displays[section.name] = Display(section.name, box2)

        return displays

    def detect(self) -> Result:
        threshold_image = self.__preprocess_image()

        displays = self.__detect_displays(threshold_image)

        ctx = self.ctx
        if ctx.options.debug:
            if displays["POWER"]:
                debug_displays(ctx, {key: disp.rect for key, disp in displays.items()})

        return None
