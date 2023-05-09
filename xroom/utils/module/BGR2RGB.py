from .FlowModule import FlowModule
import cv2
import numpy as np

class BGR2RGB(FlowModule):
    
    def __init__(self):
        pass
    
    def forward(self, images):
        images_ = []
        for image in images:
            image = np.copy(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_.append(image)
        return [images_]