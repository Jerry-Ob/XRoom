from .FlowModule import FlowModule
from utils import improc
import numpy as np

class StrokeImage(FlowModule):
    
    def __init__(self, strenth=6):
        self.strenth = strenth
        
    def forward(self, images, *args, **kwargs):
        _images = []
        for image in images:
            image = np.copy(image)
            image = improc.style.stroke(image, self.strenth)
            _images.append(image)
        return [_images]