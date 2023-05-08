def xywh2xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]