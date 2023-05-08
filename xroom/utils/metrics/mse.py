import cv2
import numpy as np

def mse(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image2.shape[0] * image1.shape[1])
    return err