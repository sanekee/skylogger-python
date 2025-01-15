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
        
class _Debug:
    def __init__(self, image: cv2.Mat, path: str, index: int):
        self._image = image.copy()
        self.__index = index
        self.__path = path

    def _write(self):
        img_path = os.path.join(self.__path, f'diag_{self.__index}.png')
        cv2.imwrite(img_path, self._image)

class FrameContext:
    def __init__(self, name: str, image: cv2.Mat, options: Options, debug_path: str):
        self.name = name
        self.options = options
        self.image = image

        if self.options.debug:
            self.__debug_dir = os.path.join(debug_path, name)
            os.makedirs(self.__debug_dir, exist_ok=True)
            self.__debugs : list[_Debug] = []
            self._new_debug()

    def __del__(self):
        self.__write_debugs()

    def __write_debugs(self):
        for _, debug in enumerate(self.__debugs):
            debug._write()

    def _new_debug(self) -> cv2.Mat:
        self.__debugs.append(_Debug(self.image, self.__debug_dir, len(self.__debugs) + 1))
        return self._get_debug_image()
    
    def _get_debug_image(self) -> cv2.Mat:
        return self.__debugs[len(self.__debugs) - 1]._image

    def _write_step(self, path: str, filename: str, image: cv2.Mat):
        output_dir = os.path.join(self.__debug_dir, path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)



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



