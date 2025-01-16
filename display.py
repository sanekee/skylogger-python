
from typing import Tuple

import cv2
from context import FrameContext
from debug import _debug
from ssd import SSD
from utils import Rect

class Digit:
    def __init__(self, ctx: FrameContext, name: str, index: int, rect: Rect):
        self.ctx = ctx
        self.name = name
        self.index = index
        self.rect = rect

        self.sliding = False
        self.max_width = 0
        self.__extract_image()
        

    def __extract_image(self):
        self.__image = self.rect.extract_image(self.ctx.image)

    def fix_size(self, width: int, height: int):
        orig_rect = self.rect
        
        # for x, just expand left
        newx = orig_rect.x - (width - orig_rect.w)
        newy = orig_rect.y

        if orig_rect.h <= 0.5 * height:
            newx = int(orig_rect.x - (width - orig_rect.w) / 2)
            newy = int(orig_rect.y - (height - orig_rect.h) / 2)
        
        elif orig_rect.h >= height * 0.8 and \
            orig_rect.w <= width * 0.6:
            # case '1'
            newy = int(orig_rect.y - (height - orig_rect.h) / 2)

        new_rect = Rect([newx, newy, width, height])
        # self.__image = self.ctx.image.copy()[new_rect.y:new_rect.y+new_rect.h, new_rect.x:new_rect.x+new_rect.w]
        self.__image = new_rect.extract_image(self.ctx.image)
        self.rect = new_rect

    def detect(self) -> str:
        res = SSD().detect(self.ctx, self.name, self.index, self.__image)
        if not res and self.sliding:
            x, y, w, h = self.rect.to_list()
            while x <= (self.max_width - w):
                newDigit = Digit(self.ctx, self.name, self.index, Rect([x, y, w, h]))
                res = newDigit.detect()
                if res is None:
                    x += 2
                    continue
                break

        return res

class Display:
    def __init__(self, ctx: FrameContext, name: str, rect: Rect, digits: list[Digit]):
        self.ctx = ctx
        self.name = name
        self.rect = rect
        self.digits  = digits
        self.gap_ratio = 0.2333
        self.fix_colon = False
        self.skip_detect = False
        
        self.__extract_image()

    def __extract_image(self):
        self.__image = self.rect.extract_image(self.ctx.image)

    def get_max_digit_size(self) -> Tuple[int, int]:
        max_w = 0
        max_h = 0
        for digit in self.digits:
            max_w = max(digit.rect.w, max_w)
            max_h = max(digit.rect.h, max_h)
        return max_w, max_h

    def __fix_colon_issue(self, width: int, height: int):

        # remove the small dots contour
        new_digits: list[Rect] = []
        for digit in self.digits:
            if digit.rect.w < 0.5 * width and \
                digit.rect.h < 0.5 * height:
                continue

            digit.index = len(new_digits)
            new_digits.append(digit)

            self.digits = new_digits

        if len(self.digits) == 3: # case where 01:23 1:2 merged in a digit zone
            _debug(self.ctx, lambda: print(f'{self.ctx.name}-{self.name}: splitting digit'))

            new_digits: list[Rect] = [self.digits[0]]
            digit = self.digits[1]

            # digit 1
            w = width
            h = height
            x = self.digits[0].rect.x2() + int(self.gap_ratio * width)
            y = digit.rect.y

            
            newDigit = Digit(self.ctx, self.name, 1, Rect([x, y, w, h]))
            newDigit.sliding = True
            newDigit.max_width = digit.rect.x2() - width

            new_digits.append(newDigit)
            
            # digit 2
            w = width
            h = height
            x = digit.rect.x2() - w
            y = digit.rect.y

            newDigit = Digit(self.ctx, self.name, 2, Rect([x, y, w, h]))
            new_digits.append(newDigit)
           
            self.digits[2].index = 3
            new_digits.append(self.digits[2])

            self.digits = new_digits

        elif len(self.digits) == 4: # case where 01:23 1: with colon

            digit = self.digits[1]

            # if not -
            if digit.rect.h >= 0.8 * height:
                _debug(self.ctx, lambda: print(f'{self.ctx.name}-{self.name}: fixing digit 1 width'))

                new_digits: list[Rect] = [self.digits[0]]

                # digit 1
                w = width
                h = height
                x = self.digits[0].rect.x2() + int(self.gap_ratio * width)
                y = digit.rect.y

                if digit.rect.h >= height * 0.8 and \
                    digit.rect.w <= width * 0.5:
                    # case '1'
                    y = int(digit.rect.y - (height - digit.rect.h) / 2)
                
                newDigit = Digit(self.ctx, self.name, 1, Rect([x, y, w, h]))
                newDigit.sliding = True
                newDigit.max_width = digit.rect.x2() - width

                new_digits.append(newDigit)
                new_digits.append(self.digits[2])
                new_digits.append(self.digits[3])
                
                self.digits = new_digits

    def fix_digits_size(self, width: int, height: int):
        if self.skip_detect:
            return
        elif self.fix_colon:
            self.__fix_colon_issue(width, height)

        for digit in self.digits:
            digit.fix_size(width, height)

        # fix self
        new_x, new_y, new_w, new_h = self.rect.to_list()
        new_x2 = new_x + new_w
        new_y2 = new_y + new_h
        for digit in self.digits:
            dx, dy, dw, dh = digit.rect.to_list()
            new_x = min(new_x, dx)
            new_y = min(new_y, dy)
            new_x2 = max(new_x2, dx + dw) 
            new_y2 = max(new_y2, dy + dh) 

        self.rect = Rect([new_x, new_y, new_x2 - new_x, new_y2 - new_y])
        self.__extract_image()

        def __debug_fix():
            img = self.__image.copy()
            [cv2.rectangle(img, d.rect.offset(self.rect).to_list(), (255,255,0), 1) for d in self.digits]
            self.ctx._write_step(f'{self.name}-fix', img)

        _debug(self.ctx, lambda: __debug_fix())
    
    def detect(self) -> str:
        res_str = ''

        for digit in self.digits:
            res = digit.detect()
            res_str += res if res is not None else ' '
        
        return res_str

