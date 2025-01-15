
import math
from typing import List, Optional
from cv2 import Mat
import cv2
from aoi import find_aoi, group_aoi
from context import FrameContext
from debug import _debug_boxes, _debug_displays, _debug_projection
from display import Digit, Display
from utils import calculate_projection, find_central_box_index, find_projection_rect_index


class Section:
    def __init__(self, name: str, angle: float, length: float):
        self.name = name
        self.angle = angle
        self.length = length

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
            "TEMPERATURE": Section("TEMPERATURE", -149.85, 4.91),
            "PROFILE": Section("PROFILE", -51.16, 2.92),
            "POWER": Section("POWER", 0, 0),
            "FAN": Section("FAN", 0.0, 4.67),
            "TIME": Section("TIME", 165.21, 4.48),
            "MODE_PREHEAT": Section("MODE_PREHEAT", 115.20, 5.0), ## recheck
            "MODE_ROAST": Section("MODE_ROAST", 84.61, 4.08),
            "MODE_COOL": Section("MODE_COOL", 59.01, 51.38), ## recheck
        }

    def __preprocess_image(self) -> Mat:
        ctx = self.ctx
        gray_image = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # Adjust size
        dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

        _, threshold_image = cv2.threshold(dilated_image, 200, 255, cv2.THRESH_BINARY) 

        return threshold_image

    def __detect_displays(self, threshold_image) -> List[Display]:
        aois = find_aoi(threshold_image, 30)

        if not aois or len(aois) == 0:
            return None

        displays: dict[str, Display]= {}

        cidx = find_central_box_index([aoi.rect for aoi in aois])
        aoi = aois[cidx]

        displays['POWER'] = Display(self.ctx, 'POWER', aoi.rect, [Digit(self.ctx, 'POWER', i, rect) for i, rect in enumerate(aoi.items)])
        
        rects = [aoi.rect for aoi in aois]

        if self.ctx.options.debug:
            _debug_projection(self.ctx, rects)

        for section in self.__sections.values():
            if section.name == 'POWER':
                continue

            pt_check = calculate_projection(aoi.rect, section.length, section.angle)
            idx2 = find_projection_rect_index(pt_check, rects)

            if idx2 is None:
                continue

            aoi2 = aois[idx2]
            displays[section.name] = Display(self.ctx, section.name, aoi2.rect, [Digit(self.ctx, section.name, i, rect) for i, rect in enumerate(aoi2.items)])

        return displays

    def detect(self) -> Optional[Result]:
        processed_image = self.__preprocess_image()

        displays = self.__detect_displays(processed_image)

        if not displays:
            return None

        if not displays["POWER"]:
            return None

        ctx = self.ctx
        if ctx.options.debug:
            _debug_displays(ctx, {key: disp.rect for key, disp in displays.items()})
            ctx._new_debug()
                
        for display in displays.values():
            print(f'{display.name}: {display.detect()}')

        return None
