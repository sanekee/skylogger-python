
import cv2
from context import FrameContext
from debug import _debug
from utils import Rect

class AOI:
    def __init__(self, rect: Rect):
        self.rect = rect
        self.items : list[AOI] = [rect]

    def group(self, rect: Rect):
        newX = min(self.rect.x, rect.x)
        newY = min(self.rect.y, rect.y)
        newX2 = max(self.rect.x2(), rect.x2())
        newW = newX2 - newX
        newY2 = max(self.rect.y2(), rect.y2())
        newH = newY2 - newY
        self.rect = Rect([newX, newY, newW, newH])
        self.items.append(rect)
        self.items = sorted(self.items, key=lambda item: item.x)


def find_aoi(ctx: FrameContext, image: cv2.Mat, minArea: int = 50, xThreshold: int = 100) -> list:
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

    def __debug_boxes():
        img = ctx.image.copy()
        [cv2.rectangle(img, box, (255,255,255), 1) for box in boxes]
        ctx._write_step("boxes", img)

    _debug(ctx, lambda: __debug_boxes())

    aoi_rows : list[AOI] = []
    cur_row:AOI = None
    for box in boxes:
        rect = Rect(box)

        if not cur_row:
            cur_row = AOI(rect)
        else:
            if is_same_row(cur_row.rect, rect):
                cur_row.group(rect)
            else:
                aoi_rows.append(cur_row)
                cur_row = AOI(rect)

    aoi_rows.append(cur_row)

    def __debug_rows():
        img = ctx.image.copy()
        for i, row in enumerate(aoi_rows):
            cv2.rectangle(img, row.rect.to_list(), (255,255,255), 2)
            [cv2.rectangle(img, rect.to_list(), (0,255,255), 1) for rect in row.items]
            cv2.putText(img, f'row-{i}', [row.rect.x, row.rect.y - 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
             
            for i, rect in enumerate(row.items):
                offset = 20 * i
                cv2.putText(img, f'{rect.x}, {rect.y} {rect.x2()}, {rect.y2()}', [rect.x, rect.y + offset], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        ctx._write_step("rows", img)

    _debug(ctx, lambda: __debug_rows())

    # group nearby boxes horizontally
    aois : list[AOI] = []
    for row in aoi_rows:
        rects = sorted(row.items, key=lambda b: b.x)
        
        cur_aoi: AOI = None
        for rect in rects:
            if not cur_aoi:
                cur_aoi = AOI(rect)
            else:
                if is_nearby_x(cur_aoi.rect, rect):
                    if cur_aoi.rect.overlapped(rect) > 0.8:
                        _debug(ctx, lambda: print('skip overlapped rect'))
                    else:
                        cur_aoi.group(rect)
                else:
                    aois.append(cur_aoi)
                    cur_aoi = AOI(rect)

        aois.append(cur_aoi)

    def __debug_aois():
        img = ctx.image.copy()
        for i, aoi in enumerate(aois):
            cv2.rectangle(img, aoi.rect.to_list(), (255,255,255), 2)
            [cv2.rectangle(img, rect.to_list(), (0,255,255), 1) for rect in aoi.items]
            cv2.putText(img, f'aoi-{i}', [aoi.rect.x, aoi.rect.y - 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        ctx._write_step("aois", img)

    _debug(ctx, lambda: __debug_aois())

    return aois
