
from typing import Optional
import cv2
from aoi import find_aoi
from context import FrameContext
from debug import _debug, _debug_displays, _debug_projection
from display import Digit, Display
from utils import calculate_projection, find_central_box_index, find_projection_rect_index

class Section:
    def __init__(self, name: str, angle: float, length: float, skip_detect: bool = False):
        self.name = name
        self.angle = angle
        self.length = length
        self.skip_detect = skip_detect

class Result:
    def __init__(self, name: str):
        self.name = name
        self.temperature = 0
        self.profile = ""
        self.power = 0
        self.fan = 0
        self.time = 0
        self.mode = ""
        
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
            "MODE_PREHEAT": Section("MODE_PREHEAT", 113.12, 4.24, True),
            "MODE_ROAST": Section("MODE_ROAST", 84.61, 4.08, True),
            "MODE_COOL": Section("MODE_COOL", 54.85, 4.77, True),
        }

    def __preprocess_image(self) -> cv2.Mat:
        ctx = self.ctx
        gray_image = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) 
        dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

        _, threshold_image = cv2.threshold(dilated_image, 200, 255, cv2.THRESH_BINARY) 

        return threshold_image

    def __detect_displays(self, threshold_image) -> list[Display]:
        aois = find_aoi(self.ctx, threshold_image, 100)

        if not aois or len(aois) == 0:
            return None

        displays: dict[str, Display]= {}

        cidx = find_central_box_index([aoi.rect for aoi in aois])
        aoi = aois[cidx]

        displays['POWER'] = Display(self.ctx, 'POWER', aoi.rect, [Digit(self.ctx, 'POWER', i, rect) for i, rect in enumerate(aoi.items)])
        
        rects = [aoi.rect for aoi in aois]
        
        _debug(self.ctx, lambda: _debug_projection(self.ctx, rects))

        for section in self.__sections.values():
            if section.name == 'POWER':
                continue

            pt_check = calculate_projection(aoi.rect.projected(), section.length, section.angle)
            idx2 = find_projection_rect_index(pt_check, rects)

            if idx2 is None:
                continue

            aoi2 = aois[idx2]

            display = Display(self.ctx, section.name, aoi2.rect, [Digit(self.ctx, section.name, i, rect) 
                                                                  for i, rect in enumerate(aoi2.items)])
            if section.name == "TIME":
                display.fix_colon = True

            display.skip_detect = section.skip_detect

            displays[section.name] =  display

        # fix digit size using maximum values from the more reliable displays (TEMPERATURE ,POWER and FAN)
        digit_width, digit_height = 0, 0
        for name in ["TEMPERATURE", 'POWER', 'FAN']:
            if not name in displays:
                continue

            display = displays[name]
            display_digit_width, display_digit_height = display.get_max_digit_size()
            digit_height = max(digit_height, display_digit_height) 
            digit_width = max(digit_width, display_digit_width) 

        for display in displays.values():
            display.fix_digits_size(digit_width, digit_height)

        return displays

    @staticmethod
    def __parse_time(time_str: str) -> int:
        if time_str == "----":
            return 0

        if len(time_str) != 4 or not time_str.isdigit():
            raise ValueError(f'invalid time {time_str}')
        
        minutes = int(time_str[:2])
        seconds = int(time_str[2:])

        if minutes >= 60:
            raise ValueError(f'invalid minutes {minutes}')
        
        if seconds >= 60:
            raise ValueError(f'invalid seconds {seconds}')
        
        total_seconds = minutes * 60 + seconds
        return total_seconds

    def detect(self) -> Optional[Result]:
        processed_image = self.__preprocess_image()

        displays = self.__detect_displays(processed_image)

        if not displays:
            print('skywalker display not found')
            return None

        if not 'POWER' in displays:
            print('skywalker power display not found')
            return None
        
        _debug(self.ctx, lambda: _debug_displays(self.ctx, {key: disp.rect for key, disp in displays.items()}))
                
        res:Result = Result(self.ctx.name)
        for display in displays.values():
            if not display.skip_detect:
                value = display.detect()
                _debug(self.ctx, lambda: print(f'{self.ctx.name}-{display.name}: {value}'))

            try:
                match display.name:
                    case "TEMPERATURE":
                        res.temperature = int(value)
                    case "POWER":
                        res.power = int(value)
                    case "FAN":
                        res.fan = int(value)
                    case "TIME":
                        res.time = SkyWalker.__parse_time(value)
                    case "PROFILE":
                        res.profile = value
                    case "MODE_PREHEAT" | "MODE_ROAST" | "MODE_COOL":
                        res.mode = display.name.removeprefix('MODE_')

            except ValueError as e:
                print(f'{self.ctx.name} - {display.name} failed to convert result ({value}): {e}')

        return res
