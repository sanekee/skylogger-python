
from typing import Optional, Tuple
import cv2

from context import FrameContext


class Rect:
    def __init__(self, rect:list):
        self.x = rect[0]
        self.y = rect[1]
        self.w = rect[2]
        self.h = rect[3]

    def x2(self) -> int:
        return self.x + self.w

    def y2(self) -> int:
        return self.y + self.h

    def area(self) -> int:
        return self.w * self.h

    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)
    
    def offset(self, rect):
        return Rect([self.x - rect.x, self.y - rect.y, self.w, self.h])

    def contains(self, rect):
        return self.x <= rect.x and rect.x2() <= self.x2() and \
                self.y <= rect.y and rect.y2() <= self.y2()
    
    def overlapped(self, rect) -> float:
        x1, y1, w1, h1 = self.to_list()
        x2, y2, w2, h2 = rect.to_list()

        max_x = max(x1, x2)
        min_x2 = min(self.x2(), rect.x2())
        max_y = max(y1, y2)
        min_y2 = min(self.y2(), rect.y2())

        if min_x2 <= max_x or min_y2 <= max_y:
            return 0.0 

        inter_area = (min_x2 - max_x) * (min_y2 - max_y)

        area1 = w1 * h1
        area2 = w2 * h2

        overlap_percentage = inter_area / min(area1, area2)

        return overlap_percentage

    def to_list(self) -> list:
        return [self.x, self.y, self.w, self.h]

    def __eq__(self, other) -> bool:
        return self.x == other.x and \
                self.y == other.y and \
                self.w == other.w and \
                self.h == other.h
    
    def projected(self):
        wmax = max(self.w, self.h * 2)
        xmin = self.x + self.w - wmax
        xmin = min(xmin, self.x)
        return Rect([xmin ,self.y, wmax, self.h])
    
    def extract_image(self, image: cv2.Mat) -> Optional[cv2.Mat]:
        image_height, image_width, _ = image.shape

        if self.x >= image_width or \
            self.y >= image_height:
            return None
        
        w = min(self.w, image_width)
        h = min(self.h, image_height)

        return image[self.y:self.y+h, self.x:self.x+w].copy()
            


    
class AOI:
    def __init__(self, rect: Rect):
        self.rect = rect
        self.items : list[AOI] = [rect]

    def merge(self, rect: Rect):
        newX = min(self.rect.x, rect.x)
        newY = min(self.rect.y, rect.y)
        newX2 = max(self.rect.x2(), rect.x2())
        newW = newX2 - newX
        newY2 = max(self.rect.y2(), rect.y2())
        newH = newY2 - newY
        self.rect = Rect([newX, newY, newW, newH])
        self.items.append(rect)
        self.items = sorted(self.items, key=lambda item: item.x)


def find_aoi(ctx: FrameContext, image: cv2.Mat, minArea: int = 50, xThreshold: int = 100, yThreshold: int = 100) -> list:
    def filter_area(contours):
        for c in contours:
            if cv2.contourArea(c) > minArea:
                yield c

    contours, _ = cv2.findContours( 
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    contours = filter_area(contours)

    boxes = [cv2.boundingRect(c) for c in contours]
    
    def is_same_row(rect1: Rect, rect2: Rect) -> bool:
        return (rect2.y <= rect1.y and rect1.y <= rect2.y2()) or \
            (rect1.y <= rect2.y and rect2.y <= rect1.y2())

    def is_nearby_x(rect1: Rect, rect2: Rect) -> bool:
        return abs(rect1.x2() - rect2.x) <= xThreshold or \
            abs(rect2.x2() - rect1.x) <= xThreshold
    
    # group into same row
    boxes = sorted(boxes, key=lambda b: b[1])  

    if ctx.options.debug:
        img = ctx.image.copy()
        [cv2.rectangle(img, box, (255,255,255), 1) for box in boxes]
        ctx._write_step("aoi-box", img)

    aoiRows : list[AOI] = []
    cur_row:AOI = None
    for box in boxes:
        rect = Rect(box)

        if not cur_row:
            cur_row = AOI(rect)
        else:
            if is_same_row(cur_row.rect, rect):
                cur_row.merge(rect)
            else:
                aoiRows.append(cur_row)
                cur_row = AOI(rect)

    aoiRows.append(cur_row)

    if ctx.options.debug:
        img = ctx.image.copy()
        for i, row in enumerate(aoiRows):
            cv2.rectangle(img, row.rect.to_list(), (255,255,255), 2)
            [cv2.rectangle(img, rect.to_list(), (0,255,255), 1) for rect in row.items]
            cv2.putText(img, f'row-{i}', [row.rect.x, row.rect.y - 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
             
            for i, rect in enumerate(row.items):
                offset = 20 * i
                cv2.putText(img, f'{rect.x}, {rect.y} {rect.x2()}, {rect.y2()}', [rect.x, rect.y + offset], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        ctx._write_step("aoi-row", img)

    # merge nearby boxes horizontally
    aois : list[AOI] = []
    for row in aoiRows:
        rects = sorted(row.items, key=lambda b: b.x)
        
        cur_aoi: AOI = None
        for rect in rects:
            if not cur_aoi:
                cur_aoi = AOI(rect)
            else:
                if is_nearby_x(cur_aoi.rect, rect):
                    if cur_aoi.rect.overlapped(rect) > 0.8:
                        print('skip overlapped rect')
                    else:
                        cur_aoi.merge(rect)
                else:
                    aois.append(cur_aoi)
                    cur_aoi = AOI(rect)

        aois.append(cur_aoi)

    if ctx.options.debug:
        img = ctx.image.copy()
        for i, aoi in enumerate(aois):
            cv2.rectangle(img, aoi.rect.to_list(), (255,255,255), 2)
            [cv2.rectangle(img, rect.to_list(), (0,255,255), 1) for rect in aoi.items]
            cv2.putText(img, f'row-{i}', [row.rect.x, row.rect.y - 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        ctx._write_step("aois", img)

    return aois
