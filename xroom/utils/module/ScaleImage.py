from .FlowModule import FlowModule
import cv2
import numpy as np

class ScaleImage(FlowModule):
    
    def __init__(self, rate=0.5):
        self.rate = rate
        
    def forward(self, image, *args, **kwargs):
        image = np.copy(image)
        image = cv2.resize(image,
                   [int(image.shape[1]*self.rate), int(image.shape[0]*self.rate)])
        return [image]