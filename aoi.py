
from typing import Tuple
import cv2


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

def group_aoi(boxes: list[list], xthreshold :int = 10) -> list[AOI]:
    boxes = sorted(boxes, key=lambda box: box[1])  
    aois : list[AOI] = []
    
    def is_overlapped(rect1: Rect, rect2: Rect)->bool:
        return (rect2.y - rect1.y) <= ythreshold and (rect2.x - rect1.x) <= xthreshold

    cur_aoi :AOI = None
    for box in boxes:
        newRect = Rect(box)
        x, y, w, h = box
        x2 = x + w
        y2 = y + h

        if not cur_aoi:
            cur_aoi= AOI(newRect)
            continue

        prev_x2 = cur_aoi.rect.x2()
        prev_y2 = cur_aoi.rect.y2()
        if is_overlapped(cur_aoi.rect, newRect):
            cur_aoi.merge(newRect)
        else:
            aois.append(cur_aoi)
            cur_aoi = AOI(newRect)

    return aois

def find_aoi(image: cv2.Mat, xThreshold: int = 100, yThreshold: int = 100) -> list:
    def filter_area(contours, min):
        for c in contours:
            if cv2.contourArea(c) > min:
                yield c

    contours, _ = cv2.findContours( 
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    contours = filter_area(contours, 50)

    boxes = [cv2.boundingRect(c) for c in contours]
    
    def is_same_row(rect1: Rect, rect2: Rect) -> bool:
        return (rect2.y <= rect1.y and rect1.y <= rect2.y2()) or \
            (rect1.y <= rect2.y and rect2.y <= rect1.y2())

    def is_nearby_x(rect1: Rect, rect2: Rect) -> bool:
        return abs(rect1.x2() - rect2.x) <= xThreshold or \
            abs(rect2.x2() - rect1.x) <= xThreshold

    # group into same row
    boxes = sorted(boxes, key=lambda b: b[1])  
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

    # merge nearby boxes horizontally
    aois : list[AOI] = []
    cur_aoi: AOI = None
    for row in aoiRows:
        rects = sorted(row.items, key=lambda b: b.x)
        
        for rect in rects:
            if not cur_aoi:
                cur_aoi = AOI(rect)
            else:
                if is_nearby_x(cur_aoi.rect, rect):
                    cur_aoi.merge(rect)
                else:
                    aois.append(cur_aoi)
                    cur_aoi = AOI(rect)

    aois.append(cur_aoi)

    return aois
