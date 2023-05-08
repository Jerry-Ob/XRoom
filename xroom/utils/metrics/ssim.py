import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def ssim(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return ssim(image1, image2, win_size=7)