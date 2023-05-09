def xywh2area(xywh):
    return xywh[2]*xywh[3]

def image2wh(image):
    return (image.shape[1], image.shape[0])