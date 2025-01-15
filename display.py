
import math
from typing import Tuple

import cv2
from aoi import Rect
from context import FrameContext
from debug import _debug_boxes
from ssd import SSD

class Digit:
    def __init__(self, ctx: FrameContext, name: str, index: int, rect: list):
        self.ctx = ctx
        self.name = name
        self.index = index
        self.rect = rect

        self.__extract_image()

    def __extract_image(self):
        x, y, w, h = self.rect.to_list()
        self.__image = self.ctx.image.copy()[y:y+h, x:x+w]

        if self.ctx.options.debug:
            self.ctx._write_step(self.name, f'2-digit-{self.index}.png', self.__image)

    def detect(self) -> str:
        return SSD().detect(self.ctx, self.name, self.index, self.__image)


class Display:
    def __init__(self, ctx: FrameContext, name: str, rect: Rect, digits: list[Digit]):
        self.ctx = ctx
        self.name = name
        self.rect = rect
        self.digits  = digits
        
        self.__extract_image()

    def __extract_image(self):
        x, y, w, h = self.rect.to_list()
        self.__image = self.ctx.image.copy()[y:y+h, x:x+w]

        if self.ctx.options.debug:
            self.ctx._write_step(self.name, f'1-image.png', self.__image)
    
    def detect(self) -> str:
        res_str = ''
        for digit in self.digits:
            res = digit.detect()
            res_str += res if res is not None else ' '


        return res_str

