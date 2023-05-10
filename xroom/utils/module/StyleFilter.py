from .FlowModule import FlowModule
from utils import improc

class StyleFilter(FlowModule):
    
    def __init__(self, style='origin'):
        '''
        style = 'origin'|'gray'|'warm'|'sketch'
        '''
        self.style = style
        
    def forward(self, images, *args, **kwargs):
        if self.style == 'origin':
            return [images]
        elif self.style == 'gray':
            return [[improc.style.grayscale(image) for image in images]]
        elif self.style == 'warm':
            return [[improc.style.sepia(image) for image in images]]
        elif self.style == 'sketch':
            return [[improc.style.pencil_sketch(image) for image in images]]
        else:
            return [images]