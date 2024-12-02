import math
import os
import cv2
import numpy as np

# segment pattern
digit_segments = [
    [1, 1, 1, 1, 1, 1, 0], # 0
    [0, 1, 1, 0, 0, 0, 0], # 1
    [1, 1, 0, 1, 1, 0, 1], # 2
    [1, 1, 1, 1, 0, 0, 1], # 3
    [0, 1, 1, 0, 0, 1, 1], # 4
    [1, 0, 1, 1, 0, 1, 1], # 5 / S
    [1, 0, 1, 1, 1, 1, 1], # 6
    [1, 1, 1, 0, 0, 0, 0], # 7
    [1, 1, 1, 1, 1, 1, 1], # 8
    [1, 1, 1, 1, 0, 1, 1], # 9
    [0, 0, 0, 0, 0, 0, 0], # NULL
    [1, 1, 1, 0, 1, 1, 1], # A
    [1, 0, 0, 0, 1, 1, 0], # T
    [1, 0, 0, 1, 1, 1, 0], # C
    [0, 0, 0, 1, 1, 1, 0], # L
]

# digit mapping
digits_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', 'A', 'T', 'C', 'L']

# monitor info
panel_monitors = {
    "TEMP": {
        'num_digits': 3,    
        'index': 0,
        'digits': [
            {
                'start': 0,
                'width': 33,
            },
            {
                'start': 33,
                'width': 34,
            },
            {
                'start': 66,
                'width': 34,
            }
        ],
        'center_angle': -150.26,
        'center_ratio': 5.2
    },
    "PROFILE": {
        'num_digits': 3,    
        'index': 1,
        'digits': [
            {
                'start': 0,
                'width': 33,
            },
            {
                'start': 33,
                'width': 34,
            },
            {
                'start': 66,
                'width': 34,
            }
        ],
        'center_angle': -51.27,
        'center_ratio': 3.24
    },
    "POWER": {
        'num_digits': 3,    
        'index': 2,
        'digits': [
            {
                'start': 0,
                'width': 33,
            },
            {
                'start': 33,
                'width': 34,
            },
            {
                'start': 66,
                'width': 34,
            }
        ],
        'center_angle': 0,
        'center_ratio': 0
    },
    "FAN": {
        'num_digits': 3,    
        'index': 3,
        'digits': [
            {
                'start': 0,
                'width': 33,
            },
            {
                'start': 33,
                'width': 34,
            },
            {
                'start': 66,
                'width': 34,
            }
        ],
        'center_angle': 1.01,
        'center_ratio': 5.25
    },
    "TIME": {
        'num_digits': 4,
        'extra': 1,
        'index': 4,
        'digits': [
            {
                'start': 0,
                'width': 24,
            },
            {
                'start': 24,
                'width': 24,
            },
            {
                'start': 48,
                'width': 4,
            },
            {
                'start': 52,
                'width': 24,
            },
            {
                'start': 76,
                'width': 24,
            }
        ],
        'center_angle': 166.51,
        'center_ratio': 5.01
    },
    "MODE": {
        'num_digits': 1,
        'index': 5,
        'digits': [
            {
                'start': 0,
                'width': 100,
            },
        ]
    },
    "ROAST_MODE": {
        'num_digits': 1,
        'index': 6,
        'digits': [
            {
                'start': 0,
                'width': 100,
            },
        ],
        'center_angle': 89.02,
        'center_ratio': 4.59
    },
    "PREHEAT_MODE": {
        'num_digits': 1,
        'index': 7,
        'digits': [
            {
                'start': 0,
                'width': 100,
            },
        ],
        'center_angle': 116.5,
        'center_ratio': 5.0,
    },
    "COOL_MODE": {
        'num_digits': 1,
        'index': 8,
        'digits': [
            {
                'start': 0,
                'width': 100,
            },
        ],
        'center_angle': 60.62,
        'center_ratio': 5.38
    },
}

class MonitorData:
    def __init__(self, box, label, settings):
        self.box = box
        self.label = label
        self.settings = settings
    
def detect_ssd(context):
    """
    Detect SSD - detect skywalker roaster seven segment display from image capture

    Args:
        context (dict): a dictionary with the required data.

    """
    gray = context['gray_image']

    ((fw,fh), baseline) = cv2.getTextSize("TEST", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) 
    context['threshold'] = threshold

    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    
    def filterArea(contours, min):
        for c in contours:
            if cv2.contourArea(c) > min:
                yield c

    contours = filterArea(contours, 50)

    boxes = [cv2.boundingRect(c) for c in contours]

    rows = group(boxes)
    for ridx, row in enumerate(rows):
        output_step(context, f'1-row-{ridx}', row)
    
    rows = merge(rows)

    for ridx, row in enumerate(rows):
        output_step(context, f'2-row-{ridx}', row)
    
    monitor_boxes = [item for row in rows for item in row]

    connect_boxes(context, monitor_boxes)

    monitor_data = build_monitor_data(context, monitor_boxes)
    # if not is_panel(rows):
    #     return None
    if len(monitor_data) < 5:
        return None

    max_width = 0
    max_height = 1
    for label, data in monitor_data.items():
        settings = data.settings

        x, y, w, h = data.box
        if settings['num_digits'] != 3:
            continue

        dig_width = int(w / settings['num_digits']) + 1

        max_width = max(max_width, dig_width)
        max_height = max(max_height, h)
        
    results = {
        'FRAME': context['section'],
        'MODE': ''
    }

    idx = 0
    for label, data in monitor_data.items():
        box = data.box
        
        settings = data.settings

        if label in ['PREHEAT_MODE', 'ROAST_MODE', 'COOL_MODE']:
            value = label.removesuffix('_MODE')
            results['MODE'] = value

            cv2.rectangle(context['diag_image'], box, (0, 255, 255), 2)
            cv2.putText(context['diag_image'], f'MODE: {value}', (box[0], box[1] + box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            continue

        extra = 0
        if 'extra' in settings:
            extra = int(settings['extra'] * 0.1 * max_width)

        min_width = (settings['num_digits'] * max_width) + extra 
        min_height = max_height
        x, y, w, h = box
        monitor_box = [x, y, w, h]

        if w < min_width:
            monitor_box[0] = x - (min_width - w)
            monitor_box[2] = min_width

        if h < min_height:
            monitor_box[1] = int(y - (min_height - h) / 2)
            monitor_box[3] = min_height
            
        if monitor_box[0] < 0:
            continue

        monitor_img = extract_box(context['threshold'], monitor_box)
        output_step_image(context, f'3-monitor-{label}', monitor_img)

        monitor_img_orig = extract_box(context['image'], monitor_box)
        output_step_image(context, f'3-monitor-{label}-orig', monitor_img_orig)

        digits = extract_digits(monitor_img, settings)

        value = ''
        for didx, digit in enumerate(digits):
            if label == 'TIME' and didx == 2:
                value += ':'
                continue

            digit_image, rank, digit_rect = detect_ssd_digit(context, digit['image'])

            if digit_image is not None:
                output_step_image(context, f'3-monitor-{label}-digit-{didx}', digit_image)

                dx = monitor_box[0] + digit['rect'][0]
                dy = monitor_box[1] + digit['rect'][1]
                dw = digit['rect'][2]
                dh = digit['rect'][3]
                cv2.rectangle(context['diag_image'], [dx, dy, dw, dh], (0, 0, 255), 3)

                dx = monitor_box[0] + digit['rect'][0] + digit_rect[0]
                dy = monitor_box[1] + digit['rect'][1] + digit_rect[1]
                dw = digit_rect[2]
                dh = digit_rect[3]
                cv2.rectangle(context['diag_image'], [dx, dy, dw, dh], (0, 255, 255), 2)
            
            if rank:
                value += digits_map[rank[0]['digit']]
            else:
                if value != '':
                    value += '_'

        cv2.putText(context['diag_image'], f'{label}: {value}', (monitor_box[0], monitor_box[1] + monitor_box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        results[label] = value

    return results

def is_panel(rows):
    for row, cols in enumerate(rows):
        if row == 0:
            if len(cols) != 2:
                return False
            
        elif row == 1:
            row_0 = rows[0]
            if len(cols) != 2:
                return False
            if not (cols[0][0] > row_0[0][0] and cols[0][0] < row_0[1][0]):
                return False
                
        elif row == 2:
            row_0 = rows[0]

            if len(cols) != 1:
                return False
            if not (cols[0][0] < row_0[0][0]):
                return False

    return True


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


def extract_box(image, box):
    x, y, w, h = box
    roi = image.copy()[y:y+h, x:x+w]

    return roi

def output_step(context, name, boxes):
    if not context['debug']:
        return

    output_path = context['debug_output_path']
    section = context.get('section', 'full')
    
    section_output_path = os.path.join(output_path, section)
    steps_output_path = os.path.join(section_output_path, "steps")
    os.makedirs(steps_output_path, exist_ok=True)

    img = context['image'].copy()

    [cv2.rectangle(img, box, (255,0,255), 2) for box in boxes]
    cv2.imwrite(os.path.join(steps_output_path, f'{name}.png'), img)


def output_step_image(context, name, img):
    if not context['debug']:
        return

    output_path = context['debug_output_path']
    section = context.get('section', 'full')
    
    section_output_path = os.path.join(output_path, section)
    steps_output_path = os.path.join(section_output_path, "steps")
    os.makedirs(steps_output_path, exist_ok=True)

    cv2.imwrite(os.path.join(steps_output_path, f'{name}.png'), img)


def extract_digits(image, settings):
    extracted_digits = []
    
    img_height, img_width = image.shape[:2]
    end_x = img_width - 1
    for idx, digit in enumerate(reversed(settings['digits'])):
        width = int(img_width * digit['width'] / 100)
        start_x = end_x - width
        if idx == len(settings['digits']) - 1:
            start_x = 0
            width = end_x - start_x

        digit_image = image[:, start_x:end_x]

        extracted_digits.insert(0, {
            'image': digit_image,
            'rect': [start_x, 0, width, img_height]
        })
        end_x -= width
    
    return extracted_digits

def check_segment(image, segment_mask):
    input_height, input_width = image.shape[:2]

    resized_mask = cv2.resize(segment_mask, (input_width, input_height))

    intersection = cv2.bitwise_and(image, resized_mask)
        
    maskCount = cv2.countNonZero(resized_mask)
    intCount = cv2.countNonZero(intersection)

    if maskCount == 0:
        return 0

    return intCount / maskCount

def detect_ssd_digit(context, image):
    image, rect = sanitize(image)
    if image is None:
        return None, None, rect

    segment_perc = {}
    for segment, mask in context['segment_masks'].items():
        segment_perc[segment] = check_segment(image, mask)

    prediction = []

    for digit, segs in enumerate(digit_segments):
        sum = 0
        pattern_threshold = 0.8
        pattern_match = True
        for idx, value in enumerate(segs):
            seg = chr(ord('a') + idx)
            perc = segment_perc[seg]
            if value == 1:
                sum = sum +perc 
                if perc < pattern_threshold:
                    pattern_match = False
            else:
                sum = sum + (1 - perc) 
                if perc >= pattern_threshold:
                    pattern_match = False

        if pattern_match:
            sum += 1
        else:
            sum -= 1

        prediction.append({
            'digit': digit,
            'sum': sum,
            'pattern': pattern_match
        })

    rank = sorted(prediction, key=lambda item: item['sum'], reverse=True)

    return image, rank, rect

def sanitize(image):
    contours, _ = cv2.findContours( 
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    if not contours:
        return None, [0, 0, image.shape[1], image.shape[0]]

    cboxes = [cv2.boundingRect(c) for c in contours]
    boxes = []
    for box in cboxes:
        if box[2] * box[3] >= 10 and not (box[0] == 0 and box[2] < 0.2 * image.shape[1]):
            boxes.append(box)

    if not boxes:
        return None, [0, 0, image.shape[1], image.shape[0]]
        
    boxes_x = sorted(boxes, key=lambda b: b[0]) 
    boxes_xw = sorted(boxes, key=lambda b: b[0]+ b[2]) 
    boxes_y = sorted(boxes, key=lambda b: b[1])
    boxes_yh = sorted(boxes, key=lambda b: b[1] + b [3]) 
    new_x = min(boxes_x[0][0], int(0.20 * image.shape[1]))
    new_y = min(boxes_y[0][1], int(0.20 * image.shape[0]))
    new_w = max(boxes_xw[-1][0] + boxes_xw[-1][2], int(0.80 * image.shape[1])) - new_x
    new_h = max(boxes_yh[-1][1] + boxes_yh[-1][3], int(0.80 * image.shape[0])) - new_y

    return image.copy()[new_y:new_y+new_h, new_x:new_x+new_w], [new_x, new_y, new_w, new_h]

def rect_center(r) -> list:
    x, y, w, h = r
    wmax = max(w, h * 2)
    xmin = x + w - wmax
    xmin = min(xmin, x)
    return (xmin + wmax // 2, y + h // 2)

def find_central_box_index(boxes):
    centers = np.array([rect_center(box) for box in boxes])

    centroid = np.mean(centers, axis=0)

    distances = np.linalg.norm(centers - centroid, axis=1)
    closest_index = np.argmin(distances)
    return closest_index


def calculate_p2(box1, ratio, angle):
    x1, y1, w1, h1 = box1
    px, py = rect_center(box1) 

    wmax = max(w1, h1*2)
    xmin = x1 + w1 - wmax
    xmin = min(x1, xmin)
    angle_radians = math.radians(angle)
    dx = h1 * ratio * math.cos(angle_radians)
    dy = h1 * ratio * math.sin(angle_radians)
    return (int(px + dx), int(py + dy))

def find_box2(pt2, boxes):
    px, py = pt2 
    for box in boxes:
        x, y, w, h = box
        wmax = max(w, h*2)
        xmin = x + w - wmax
        xmin = min(x, xmin)
        if xmin <= px <= xmin + wmax and \
            y <= py <= y + h:
            return box

    return None


def build_monitor_data(context, boxes) -> dict[str, MonitorData]:
    ret = {}
    cidx = find_central_box_index(boxes)
    box1 = boxes[cidx]

    x1,y1,w1,h1 = box1
    center1 = rect_center(box1)

    ret['POWER'] = MonitorData(box1, 'POWER', panel_monitors['POWER'])
    
    for k, settings in panel_monitors.items():
        if k == 'POWER':
            continue

        if not 'center_ratio' in settings.keys() or \
            not 'center_angle' in settings.keys():
            continue

        pt_check = calculate_p2(box1, settings['center_ratio'], settings['center_angle'])

        box2 = find_box2(pt_check, boxes)

        if box2 is None:
            continue

        x2, y2, w2, h2 = box2
        center2 = rect_center(box2) 

        cv2.line(context['diag_image'], center1, center2, (0, 255, 0), 2)
        cv2.line(context['diag_image'], center1, pt_check, (0, 255, 255), 1)

        # Calculate details
        line_length = int(math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2))
        ratio = round(line_length / h1, 2)
        angle = round(calculate_angle(center1, center2), 2)

        # Text to display
        text = f"L:{line_length}, R:{ratio}, A:{angle}"

        # Calculate midpoint and put text
        mid_pt = midpoint(center1, center2)
        cv2.putText(context['diag_image'], text, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        ret[k] = MonitorData(box2, k, settings)

    return ret

def calculate_angle(pt1, pt2):
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    return math.degrees(math.atan2(delta_y, delta_x))

def midpoint(pt1, pt2):
    return ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

def connect_boxes(context, boxes):
    cidx = find_central_box_index(boxes)
    box1 = boxes[cidx]

    x1,y1,w1,h1 = box1
    center1 = rect_center(box1)
    
    for i, box in enumerate(boxes):
        if i == cidx:
            continue

        x2, y2, w2, h2 = box
        center2 = rect_center(box) 

        cv2.line(context['diag_image'], center1, center2, (188, 255, 188), 1)

        # Calculate details
        line_length = int(math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2))
        ratio = round(line_length / h1, 2)
        angle = round(calculate_angle(center1, center2), 2)

        # Text to display
        text1 = f"L:{line_length}, R:{ratio}, A:{angle}"
        text2 = f"Pt2:{center2[0]},{center2[1]}"

        # Calculate midpoint and put text
        mid_pt = midpoint(center1, center2)
        cv2.putText(context['diag_image'], text1, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (188, 255, 255), 1)
        cv2.putText(context['diag_image'], text2, (mid_pt[0], mid_pt[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (188, 255, 255), 1)
