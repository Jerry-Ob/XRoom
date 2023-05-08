import numpy as np

def iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))

    area1 = (xmax1 - xmin1 ) * (ymax1 - ymin1) 
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    iou = inter_area / (area1 + area2 - inter_area )
    return iou