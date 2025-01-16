import csv
import os
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
        
class FrameContext:
    def __init__(self, name: str, image: cv2.Mat, options: Options, debug_path: str):
        self.name = name
        self.options = options
        self.image = image

        self.__step_counter = 1

        if self.options.debug:
            self.__debug_dir = os.path.join(debug_path, name)
            os.makedirs(self.__debug_dir, exist_ok=True)

    def _write_step(self, filename: str, image: cv2.Mat):
        if not self.options.debug:
            return

        output_path = os.path.join(self.__debug_dir, f'{self.__step_counter}-{filename}.png')
        cv2.imwrite(output_path, image)
        self.__step_counter += 1

    
class Context:
    def __init__(self, args: argparse.Namespace):
        self.settings = Settings(args.input_path, args.output_path)
        self.options = Options(args)

        os.makedirs(self.settings.output_path, exist_ok=True)

        if self.options.debug:
            self._debug_path = os.path.join(self.settings.output_path, '_debug')
        

    def new_frame_context(self, name: str, image: cv2.Mat):
        return FrameContext(name, image, self.options, self._debug_path)
