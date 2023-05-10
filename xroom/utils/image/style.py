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

def grayscale(image):
    image = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def sepia(image):
    img = np.copy(image)
    img_sepia = np.array(img, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]]))
    img_sepia[np.where(img_sepia > 255)] = 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia

def pencil_sketch(image):
    img = np.copy(image)
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_color
 
 