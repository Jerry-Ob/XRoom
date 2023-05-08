import streamlit as st
import cv2

from utils import *

path_video = '../video/sample-3.mp4'

flow_model = FlowModuleList([
    BoardTracker(1, transition=0.95, drop_thres=0.97),
    FlowModuleBranch({
        'enhanced': ImageEnhance(sharpen=0.1),
        'stroke': StrokeImage(strenth=6)
    }),
])

box_test = st.empty()
box_enhance = st.empty()

cap = cv2.VideoCapture(path_video)
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None or not ret:
        break
    
    flow = flow_model.forward(frame)
    
    box_test.image(flow['enhanced'][0][0])
    box_enhance.image(flow['stroke'][0][0])
    