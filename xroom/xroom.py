import streamlit as st
import cv2

from utils import *

path_video = '../video/sample-3.mp4'

board_flow = FlowModuleList([
    BoardTracker(2, transition=0.9, drop_thres=0.97),
    FlowModuleBranch({
        'enhanced': ImageEnhance(sharpen=0.1),
        'stroke': StrokeImage(strenth=6),
        'ocr': FlowModuleList([
            StrokeImage(strenth=6),
            OpticalCharRecognition(lang=['en'], gpu=False, refresh_thres=800)
        ])
    }),
])

people_flow = FlowModuleList([
    PeopleTrack(model_scale='n', gpu=False),
    SuperResolution(weights='./weights/drrn_B1U9.pth', residual_layers=9, gpu=False)
])

flow_model = FlowModuleBranch({
    'people': people_flow,
    'board': board_flow
})

box_people = st.empty()
box_test = st.empty()
box_enhance = st.empty()
box_text = st.empty()

cap = cv2.VideoCapture(path_video)
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None or not ret:
        break
    
    flow = flow_model.forward(frame)
    board = flow.get('board')
    people = flow.get('people')
    
    box_test.image(board['enhanced'][0][0])
    box_enhance.image(board['stroke'][0][0])
    box_text.write(board['ocr'][0][0])
    
    for person in people[0]:
        box_people.image(person)
    