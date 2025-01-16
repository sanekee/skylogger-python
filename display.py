
import math
from typing import Tuple

import cv2
from aoi import Rect
from context import FrameContext
from debug import _debug_boxes
from ssd import SSD
from utils import area

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
        # x, y, w, h = self.rect.to_list()
        # self.__image = self.ctx.image.copy()[y:y+h, x:x+w]
        self.__image = self.rect.extract_image(self.ctx.image)

    def fix_size(self, width: int, height: int):
        orig_rect = self.rect
        
        # issue with TIME display colon got merged into other digit
        if width < orig_rect.w:
            return

        # for x, just expand left
        newx = orig_rect.x - (width - orig_rect.w)
        newy = orig_rect.y

        if orig_rect.h <= 0.5 * height:
            newx = int(orig_rect.x - (width - orig_rect.w) / 2)
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
            print(f'{self.ctx.name}-{self.name}: splitting digit')

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
                print(f'{self.ctx.name}-{self.name}: fixing digit 1 width')

                new_digits: list[Rect] = [self.digits[0]]

                # digit 1
                w = width
                h = height
                x = self.digits[0].rect.x2() + int(self.gap_ratio * width)
                y = digit.rect.y

                
                newDigit = Digit(self.ctx, self.name, 1, Rect([x, y, w, h]))
                newDigit.sliding = True
                newDigit.max_width = digit.rect.x2() - width

                new_digits.append(newDigit)
                new_digits.append(self.digits[2])
                new_digits.append(self.digits[3])
                
                self.digits = new_digits
            
        # if self.ctx.options.debug:
        #     img = self.__image.copy()
        #     [cv2.rectangle(img, d.rect.offset(self.rect).to_list(), (255,255,0), 1) for d in self.digits]
        #     self.ctx._write_step(f'{self.name}-fix', img)

    def fix_digits_size(self, width: int, height: int):
        if self.name.startswith("MODE_"):
            return
        elif self.fix_colon:
            self.__fix_colon_issue(width, height)

        for digit in self.digits:
            digit.fix_size(width, height)

        if self.ctx.options.debug:
            img = self.__image.copy()
            [cv2.rectangle(img, d.rect.offset(self.rect).to_list(), (255,255,0), 1) for d in self.digits]
            self.ctx._write_step(f'{self.name}-fix', img)
    
    def __fix_time_digit(self, digit: Digit) -> str:
        if digit.index != 1:
            return ' '
        
        new_digits: list[Rect] = []

        res_str = ''
        if len(self.digits) == 3: # case where 01:23 1:2 merged in a digit zone
            print(f'{self.ctx.name}-{self.name}-{digit.index}: splitting digit')
            # digit 1
            w = self.__digit_size[0]
            h = self.__digit_size[1]
            x = self.digits[0].rect.x2() + int(self.gap_ratio * self.__digit_size[0])
            y = self.digits[0].rect.y

            idx = 1
            while x + w < digit.rect.x2():
                newDigit = Digit(self.ctx, self.name, idx, Rect([x, y, w, h]))
                res = newDigit.detect()
                if res is None:
                    x += 5
                    continue

                break

            new_digits.append(newDigit)
            res_str += res if res is not None else ' '
            
            # digit 2
            idx = 2
            w = self.__digit_size[0]
            h = self.__digit_size[1]
            x = digit.rect.x2() - w
            y = self.digits[0].rect.y

            newDigit = Digit(self.ctx, self.name, idx, Rect([x, y, w, h]))
            res = newDigit.detect()
            
            new_digits.append(newDigit)
           
            res_str += res if res is not None else ' '
            
            print(f'{self.ctx.name}-{self.name}-{digit.index}: fix result {res_str}')
        
        if len(self.digits) == 4: # case where colon 01:23 1: merged in a digit zone
            print(f'{self.ctx.name}-{self.name}-{digit.index}: fixing digit')
            # digit 1
            w = self.__digit_size[0]
            h = self.__digit_size[1]
            x = self.digits[0].rect.x2() + int(self.gap_ratio * self.__digit_size[0])
            y = self.digits[0].rect.y

            res_str = ''
            idx = 1
            while x + w < digit.rect.x2():
                newDigit = Digit(self.ctx, self.name, idx, Rect([x, y, w, h]))
                res = newDigit.detect()
                if res is None:
                    x += 5
                    continue
                break

            new_digits.append(newDigit)
            res_str += res if res is not None else ' '
            print(f'{self.ctx.name}-{self.name}-{digit.index}: fix result {res_str}')

        if len(new_digits) > 0 and len(self.digits) > 2:
            new_digits.insert(0, self.digits[0])
            
            for i in range(len(self.digits) - 1, len(self.digits), 1):
                new_digits.append(Digit(self.ctx, self.name, i, Rect(self.digits[i].rect.to_list())))
            

            if self.ctx.options.debug:
                img = self.__image.copy()

                for d in new_digits:
                    cv2.rectangle(img, d.rect.offset(self.rect).to_list(), (0, 255, 255), 1)

                self.ctx._write_step(f'{self.name}-fixdigit', img)

        return res_str

    def detect(self) -> str:
        res_str = ''

        for digit in self.digits:
            res = digit.detect()
            res_str += res if res is not None else ' '
        
        return res_str

