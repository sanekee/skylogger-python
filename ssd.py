
import cv2
from context import FrameContext


class SSDResults:
    def __init__(self, name: str):
        self.name = name

class SSD:
    def __init__(self, ctx: FrameContext):
        self.ctx = ctx

        self.__init_patterns()
        self.__init_zones()

    def __init_patterns(self):
        self.__patterns= {
            "0000000": None,
            "1111110": '0',
            "0110000": '1',
            "1101101": '2',
            "1111001": '3',
            "0110011": '4',
            "1011011": '5',
            "1011111": '6',
            "1110000": '7',
            "1111111": '8',
            "1111011": '9',
            "1110111": 'A',
            "1000110": 'T',
            "1001110": 'C',
            "0001110": 'L',
        }
    
    def __init_zones(self):
        self.__zones = {
            "top": lambda image: image[:int(0.3 * image.shape[0]), int(0.1 * image.shape[1]):int(0.9 * image.shape[1])],
            "top-right": lambda image: image[int(0.1 * image.shape[0]):int(0.5 * image.shape[0]), int(0.7 * image.shape[1]):],
            "bottom-right": lambda image: image[int(0.5 * image.shape[0]):int(0.9 * image.shape[0]), int(0.7 * image.shape[1]):],
            "bottom": lambda image: image[int(0.7 * image.shape[0]):, int(0.1 * image.shape[1]):int(0.9 * image.shape[1])],
            "bottom-left": lambda image: image[int(0.5 * image.shape[0]):int(0.9 * image.shape[0]), :int(0.3 * image.shape[1])],
            "top-left": lambda image: image[int(0.1 * image.shape[0]):int(0.5 * image.shape[0]), :int(0.3 * image.shape[1])],
            "middle": lambda image: image[int(0.4 * image.shape[0]):int(0.6 * image.shape[0]), int(0.1 * image.shape[1]):int(0.9 * image.shape[1])],
        }

    def __preprocess_image(self) -> cv2.Mat:
        ctx = self.ctx
        binary = cv2.adaptiveThreshold(
            ctx.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cleaned_binary

    def detect(self):
        processed_image = self.__preprocess_image()

