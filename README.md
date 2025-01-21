# Skylogger - Python, OpenCV2, PaddleOCR

## Overview

An Itop Skywalker coffee roaster logger. 

[AI Version](https://github.com/sanekee/skylogger-paddle)

Extract roasting profile using OpenCV2 image processing to recognize seven segment display digits from the coffee roaster control panel.

```mermaid
flowchart LR
    A[Start] --> B[Load Video]
    B --> C{Get Each Interval}
    C -->|No| D['Write Results] --> E[END]
    C --> |Yes| F[Extract Image]
    F --> G[Detect Area of Interest #40;AOI#41;]
    G --> I[Grouping Displays]
    I --> J[Recognize SSD]
    J --> K[Return Result] --> C
```

## Sample

![Sample](./assets/sample.png)

## Usage

1. Using Dev Container
2. runs with python3 main.py &lt;input video&gt; &lt;output path&gt;<br/>
   **Args**:
   | Arg        | Description                           |
   |------------|---------------------------------------|
   | --rotate   | Rotate image [auto,<number of degree] |
   | --interval | Extract frame every second            |
   | --skip     | Skip seconds from beginning of video  |
   | --count    | Number of frame to extract            |
   | --debug    | Output debugging images               |

   Example:
    ```shell
    python3 main.py video.mp4 output --debug=true  --rotate=auto --skip=5 --count=10 --interval=30
    ```

## Implementation

### 1. Detect Area of Interest (AOI)

![Area of Interest](./assets/step1-aoi.png)

### 2. Extract Digits

| Area of Interest | Digits                                             |
|------------------|----------------------------------------------------|
| Temperature      | ![TEMPERATURE](./assets/step2-aoi-temperature.png) |
| Power            | ![POWER](./assets/step2-aoi-power.png)             |
| Fan              | ![FAN](./assets/step2-aoi-fan.png)                 |
| Profile          | ![PROFILE](./assets/step2-aoi-profile.png)         |
| Time             | ![TIME](./assets/step2-aoi-time.png)               |

### 3. Detect Digit Segments

![Segments](./assets/step3-segments.png)


### 4. Sample output

[Result.csv](./assets/results.csv)

## What Next
- [ ] Explore detection with tesseract
- [ ] Explore detection with tensorflow / pytorch
