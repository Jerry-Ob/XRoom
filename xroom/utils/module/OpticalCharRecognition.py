from .FlowModule import FlowModule
from utils import metrics
from utils import coord
import easyocr
import cv2

class OpticalCharRecognition(FlowModule):
    
    def __init__(self, lang=['en'], gpu=False, refresh_thres=800, intime=False):
        self.reader = easyocr.Reader(lang, gpu=gpu)
        self.previous_frames = None
        self.text_list_history = []
        self.refresh_thres = refresh_thres
        self.intime = intime
        
    def forward(self, images, *args, **kwargs):
        if self.previous_frames is None or self.intime:
            self.previous_frames = images
            text_list = []
            change_mark = []
            for img in images:
                text_list.append(self.reader.readtext(img, detail=0))
                change_mark.append(True)
            self.text_list_history = text_list
            return [text_list, change_mark]
        
        else:
            indicator = [metrics.MSE(cv2.resize(images[i], coord.image2wh(self.previous_frames[i])), self.previous_frames[i])
                         for i in range(0, max(len(images), len(self.previous_frames)))]
            self.previous_frames = images
            text_list = []
            change_mark = []
            for i in range(0, len(images)):
                if len(indicator) < i or indicator[i] > self.refresh_thres:
                    text_list.append(self.reader.readtext(images[i], detail=0))
                    change_mark.append(True)
                else:
                    text_list.append(self.text_list_history[i])
                    change_mark.append(False)
            self.text_list_history = text_list
            return [text_list, change_mark]