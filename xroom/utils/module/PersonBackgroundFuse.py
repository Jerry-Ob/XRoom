from .FlowModule import FlowModule
import numpy as np
import cv2


class PersonBackgroundFuse(FlowModule):
    
    def __init__(self, background='./', occupy_rate=0.2):
        self.background = cv2.imread(background)
        self.occupy_rate = occupy_rate
        
    def _attach_person(self, person, background, rate=0.2):
        
        person = np.copy(person)
        background = np.copy(background)
        
        scale_rate = rate*background.shape[0] / person.shape[0]
        person = cv2.resize(person, [int(person.shape[1]*scale_rate), int(person.shape[0]*scale_rate)])
        w = person.shape[0]
        h = person.shape[1]
        
        x = background.shape[0] - person.shape[0]
        y = int(background.shape[1]/2 - (h/2))
        
        roi = background[x:x+w, y:y+h]
        
        img2gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 15, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        roi = cv2.bitwise_and(roi, roi, mask=mask_inv)
        dst = cv2.add(roi, person)
        background[x:x+w, y:y+h] = dst
        
        return background
        
    def forward(self, images, *args, **kwargs):
        fused_images = [self._attach_person(image,
                                            self.background,
                                            self.occupy_rate)
                        for image in images]
        return [fused_images]