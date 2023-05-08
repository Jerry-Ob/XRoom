import cv2
import numpy as np

def stroke(image, strength=6):
    image = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                                255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY,
                                15,
                                strength)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def cartoon(image, epoch=11, ksize=21, sigma_color=11, sigma_space=11):
    img_cartoon = np.copy(image)
    for _ in range(epoch):
        img_cartoon = cv2.bilateralFilter(img_cartoon,ksize,sigma_color,sigma_space)
    return img_cartoon

def canny(image, thres=0.5):
    imgGray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), thres)
    imgCanny = cv2.Canny(imgBlur,200,200)
    
    return imgCanny