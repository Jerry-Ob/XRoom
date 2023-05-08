from .FlowModule import FlowModule
from utils import improc

class ImageEnhance(FlowModule):
    
    def __init__(self, sharpen=0.1):
        self.sharpen = sharpen
    
    def forward(self, images, *args, **kwargs):
        images_ = []
        for image in images:
            image = improc.enhance.balance_light(image)
            image = improc.enhance.increase_contract(image)
            image = improc.enhance.sharpen(image, strength=self.sharpen)
            images_.append(image)
        return [images_]
        