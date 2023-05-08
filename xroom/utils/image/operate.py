import cv2

def clip_image(image, xyxy):
    return image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

def detect_shape(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    objects = []
    for obj in contours:
        perimeter = cv2.arcLength(obj,True)
        approx = cv2.approxPolyDP(obj, 0.02*perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        objects.append([x, y, w, h, approx])
    return objects
