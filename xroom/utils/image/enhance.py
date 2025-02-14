import cv2
import numpy as np

def sharpen(image, strength=0.1):
    kernel = np.array([[0, -1, 0], [-1, strength+5, -1], [0, -1, 0]], np.float32)
    return cv2.filter2D(image, -1, kernel=kernel)

def balance_light(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, image)
    return image