import csv
import os
from typing import List
import cv2
import argparse

class Settings:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        
class Options:
    def __init__(self, args: argparse.Namespace):
        self.skip = args.skip 
        self.count = args.count 
        self.interval = args.interval
        self.rotate = args.rotate
        self.debug = args.debug
        self.training = args.training
        
class FrameContext:
    def __init__(self, name: str, image: cv2.Mat, options: Options, debug_path: str):
        self.name = name
        self.options = options
        self.image = image
        self.__debug_count = 1

        if self.options.debug:
            debug_dir = os.path.join(debug_path, name)
            self._debug_path = os.path.join(debug_dir, 'diag.png')
            self._debug_image = self.image.copy()
            
            os.makedirs(debug_dir, exist_ok=True)
    
    def _write_debug(self):
        cv2.imwrite(self._debug_path, self._debug_image)

class Result:
    def __init__(self, name: str, temperature: int, profile: str, power: int, fan: int, time: int, mode: str):
        self.name = name
        self.temperature = temperature
        self.profile = profile
        self.power = power
        self.fan = fan
        self.time = time
        self.mode = mode
        

class Context:
    def __init__(self, args: argparse.Namespace):
        self.settings = Settings(args.input_path, args.output_path)
        self.options = Options(args)

        os.makedirs(self.settings.output_path, exist_ok=True)

        self._result_path = os.path.join(self.settings.output_path, 'results.csv')
        
        if self.options.debug:
            self._debug_path = os.path.join(self.settings.output_path, 'debug')
        

    def new_frame_context(self, name: str, image: cv2.Mat):
        return FrameContext(name, image, self.options, self._debug_path)

    def write_result(self, results: list[Result]):
        if len(results) == 0:
            return

        outFile = os.path.join(self.settings.output_path, 'results.csv')
        with open(outFile, 'w') as f:
            wrt = csv.writer(f, delimiter=',')
            wrt.writerow(['name', 'time', 'temperature','profile','power','fan','mode'])

            for res in results:
                wrt.writerow([res.name, res.time, res.temperature, res.profile, res.power, res.fan, res.mode])



