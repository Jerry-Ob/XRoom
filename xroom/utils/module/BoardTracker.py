from .FlowModule import FlowModule
from utils import improc
from utils import coord
from utils import metrics

class BoardTracker(FlowModule):
    
    def __init__(self, max_board=1, transition=0.95, drop_thres=0.97):
        self.max_board = max_board
        self.boards = None
        self.transition = transition
        self.drop_thres = drop_thres
        
    def _get_boards(self, image):
        imgCanny = improc.style.canny(image, 0.5)
        objects = improc.detect_shape(imgCanny)
        
        objects_size = [coord.xywh2area(obj) for obj in objects]
        objects_candidate = sorted(objects_size, reverse=True)[:self.max_board]
        
        boards = []
        for item in objects_candidate:
            index = objects_size.index(item)
            obj = objects[index]
            obj = coord.xywh2xyxy(obj)
            boards.append(obj)
        self.boards = boards
    
    
    def _track_boards(self, image, transition=0.95, drop_thres=0.97):
        imgCanny = improc.style.canny(image, 0.5)
        objects = improc.detect_shape(imgCanny)
        for i in range(0, len(self.boards)):
            ious = []
            pbox = self.boards[i]
            for obj in objects:
                ious.append(metrics.IOU(pbox, coord.xywh2xyxy(obj)))
                c_bbox = coord.xywh2xyxy(objects[ious.index(max(ious))])
                if max(ious) > drop_thres:
                    c_bbox = [int((pbox[i]*transition + c_bbox[i]*(1-transition))) for i in range(0, 4)]
                else:
                    c_bbox = pbox
            self.boards[i] = c_bbox
            
    
    def forward(self, image, *args, **kwargs):
        
        if self.boards is None:
            self._get_boards(image)
        else:
            self._track_boards(image, self.transition, self.drop_thres)
            
        board_images = [improc.clip_image(image, bbox) for bbox in self.boards]
        
        return [board_images]