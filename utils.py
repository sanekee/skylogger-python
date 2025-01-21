
import math
from typing import Optional, Tuple
import cv2
import numpy as np

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
            


def find_central_box_index(rects: list[Rect]):
    centers = np.array([rect.center() for rect in rects])

    centroid = np.mean(centers, axis=0)

    distances = np.linalg.norm(centers - centroid, axis=1)
    closest_index = np.argmin(distances)
    return closest_index


def calculate_projection(rect: Rect, ratio: float, angle: float) -> Tuple[int, int]:
    x1, y1, w1, h1 = rect.to_list()
    px, py = rect.projected().center()

    angle_radians = math.radians(angle)
    dx = h1 * ratio * math.cos(angle_radians)
    dy = h1 * ratio * math.sin(angle_radians)
    return (int(px + dx), int(py + dy))

def find_projection_rect_index(pt2: Tuple[int, int], rects: list[Rect]) -> Optional[int]:
    px, py = pt2 
    for i, rect in enumerate(rects):
        x, y = rect.projected().center()
        distance = int(math.sqrt((((px - x) ** 2) + ((py - y) ** 2))))
        if distance <= (rect.h * 2):
            return i

    return None

def calculate_angle(pt1, pt2):
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    return math.degrees(math.atan2(delta_y, delta_x))

def midpoint(pt1, pt2):
    return ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

def group(boxes, threshold = 0):
    boxes = sorted(boxes, key=lambda b: b[1])  # Sort by y

    rows = []

    def overlap_y(y1, h1, y2, h2):
        return y1 < (y2 + h2 + threshold) and y2 < (y1 + h1 + threshold)

    def overlap_x(x1, w1, x2, w2):
        return x1 < (x2 + w2 + threshold) and x2 < (x1 + w1 + threshold)

    # Group items into rows based on overlap in y and h
    current_row = []
    current_row_y = 0
    current_row_h = 0
    for idx, item in enumerate(boxes):
        x, y, w, h = item
        
        if not current_row:
            current_row.append(item)
            current_row_y = y
            current_row_h = h
        else:
            is_current = False
            if overlap_y(current_row_y, current_row_h, y, h):
                is_current = True
            else:
                for box in current_row:
                    if overlap_x(box[0], box[2], item[0], item[2]) and \
                        y - (box[1] + box[3]) <= 10:
                        is_current = True
                        break

            if is_current:
                current_row.append(item)
                current_row_y = min(current_row_y, y)
                current_row_h = max(current_row_y + current_row_h, y + h) - current_row_y
            else:
                rows.append(current_row)
                current_row = [item]
                current_row_y = y
                current_row_h = h

    if current_row:
        rows.append(current_row)

    for row in rows:
        row.sort(key=lambda item: item[0])


    return rows

def merge(rows, xthreshold = 100, ythreshold = 100):
    groups = []
    for row in rows:
        new_row = []

        row_x, row_y, row_w, row_h = [0, 0, 0, 0]
        for item in row:
            x, y, w, h = item
            x2 = x + w
            y2 = y + h

            if not new_row:
                new_row.append([x,y,w,h])
                row_x, row_y, row_w, row_h = x, y, w, h
            else:
                current_box = new_row[-1]
                prev_x2 = row_x + row_w
                prev_y2 = row_y + row_h
                if (y - row_y) <= ythreshold and (x - prev_x2) <= xthreshold:
                    current_box[2] = max((x + w), prev_x2) - current_box[0]
                    current_y = current_box[1]
                    current_box[1] = min(current_box[1], y)
                    current_box[3] = max(prev_y2, y2) - min(current_box[1], current_y)
                    row_x, row_y, row_w, row_h = current_box
                else:
                    new_row.append([x,y,w,h])
                    row_x, row_y, row_w, row_h = x, y, w, h

        groups.append(new_row)

    return groups

def merge(rows, xthreshold = 100, ythreshold = 100):
    groups = []
    for row in rows:
        new_row = []

        row_x, row_y, row_w, row_h = [0, 0, 0, 0]
        for item in row:
            x, y, w, h = item
            x2 = x + w
            y2 = y + h

            if not new_row:
                new_row.append([x,y,w,h])
                row_x, row_y, row_w, row_h = x, y, w, h
            else:
                current_box = new_row[-1]
                prev_x2 = row_x + row_w
                prev_y2 = row_y + row_h
                if (y - row_y) <= ythreshold and (x - prev_x2) <= xthreshold:
                    current_box[2] = max((x + w), prev_x2) - current_box[0]
                    current_y = current_box[1]
                    current_box[1] = min(current_box[1], y)
                    current_box[3] = max(prev_y2, y2) - min(current_box[1], current_y)
                    row_x, row_y, row_w, row_h = current_box
                else:
                    new_row.append([x,y,w,h])
                    row_x, row_y, row_w, row_h = x, y, w, h

        groups.append(new_row)

    return groups


def extract_box(image, box):
    x, y, w, h = box
    roi = image.copy()[y:y+h, x:x+w]

    return roi

def area(width: int, height: int) -> int:
    return width * height
