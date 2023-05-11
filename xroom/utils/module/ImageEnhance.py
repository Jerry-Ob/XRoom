from .FlowModule import FlowModule
from utils import improc
import numpy as np

class ImageEnhance(FlowModule):
    
    def __init__(self, sharpen=0.1):
        self.sharpen = sharpen
    
    def forward(self, images, *args, **kwargs):
        images_ = []
        for image in images:
            image = np.copy(image)
            image = improc.enhance.balance_light(image)
            image = improc.enhance.sharpen(image, strength=self.sharpen)
            images_.append(image)
        return [images_]
        