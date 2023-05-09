from .FlowModule import FlowModule
from ultralytics import YOLO
import cv2
import numpy as np

class PeopleTrack(FlowModule):
    
    def __init__(self, model='', gpu=False):
        self.model = YOLO(model)
        if gpu:
            self.model.to('cuda')
        self.previous_anchors = None
        self.previous_mask = None
    
    def forward(self, image, *args, **kwargs):
        image = np.copy(image)
        image = cv2.resize(image, [640, 384])
        labels = self.model.predict(image)
        valid_parts = []
        if len(labels) > 0 and labels[0].masks is not None:
            masks = [m.masks.numpy()[0] for m in labels[0].masks]
            anchors = [box.xyxyn.numpy()[0] for box in labels[0].boxes]
            classes = [box.cls.numpy()[0] for box in labels[0].boxes]
            previous_mask_ = []
            previous_anchors_ = []
            for id, (mask, anchor) in enumerate(zip(masks, anchors)):
                if classes[id] != 0:
                    continue
                image_ = np.copy(image)
                image_[:,:,0] = image_[:,:,0] * mask
                image_[:,:,1] = image_[:,:,1] * mask
                image_[:,:,2] = image_[:,:,2] * mask
                image_ = image_[int(anchor[1]*384):int(anchor[3]*384),
                                int(anchor[0]*640):int(anchor[2]*640)]
                valid_parts.append(image_)
                previous_mask_.append(mask)
                previous_anchors_.append(anchor)

            self.previous_mask = previous_mask_
            self.previous_anchors = previous_anchors_
                
            return [valid_parts, masks, anchors]
                
        else:
            return [valid_parts, [], []]
        