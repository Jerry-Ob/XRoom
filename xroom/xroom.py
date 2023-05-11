import streamlit as st
import cv2
import time
from utils import *
from config import *
from xroom_model import load_flow_model

########## Page Construct ###############

## Page Init 
st.set_page_config(page_title='XRoom', layout='wide', page_icon=PATH_ICON_SMALL)
hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

## Page Construct
pannel_control = st.sidebar.container()
pannel_setting = st.sidebar.container()
pannel_body = st

## Sidebar Construct
icon,_ = pannel_control.columns(2)
icon.image(PATH_ICON)

pannel_video, pannel_camera = pannel_control.tabs(['🎞️ Video', '📹 Camera'])
control_path_video = pannel_video.text_input('🎞️ Video Path', PATH_DEFAULT_VIDEO)

control_start_play = pannel_video.button('&nbsp;&nbsp;🔴&nbsp;&nbsp;Play&nbsp;&nbsp;&nbsp;&nbsp;')

pannel_camera.write('📹 Camera Preview')
control_camera_preview = pannel_camera.empty()
control_start_camera = pannel_camera.button('&nbsp;&nbsp;🔴&nbsp;&nbsp;Start&nbsp;&nbsp;&nbsp;&nbsp;')
control_camera_preview.image(PATH_PLACEHOLDER)

pannel_setting.subheader('🛠️ Preferance')
background_select, style_select = pannel_setting.columns(2)
control_background_select = background_select.selectbox(
    '🖼️ Background',
    list(background_packs.keys())
)
control_style_select = style_select.selectbox(
    '🪄 Style',
    list(style_filters.keys())
)
control_style_select = style_filters[control_style_select]
conf_background_path = background_packs[control_background_select]

background_preview = cv2.imread(conf_background_path)
preview_filter = StyleFilter(control_style_select)
background_preview = preview_filter.forward([background_preview])[0]
background_preview = background_preview[0]
background_preview = cv2.cvtColor(background_preview, cv2.COLOR_BGR2RGB)
pannel_setting.image(background_preview)


## Setting Pannel Construct

pannel_setting_more = pannel_setting.expander('⚙️ Advanced Settings')
pannel_runtime = pannel_setting.expander('⏳ Run Time')
pannel_setting_more.subheader('Board')
conf_lang_set = pannel_setting_more.multiselect('OCR Languages', list(language_map.keys()), ['English'])
conf_lang_set = [language_map.get(item) for item in conf_lang_set]
conf_ocr_refresh_thres = pannel_setting_more.slider('OCR Refresh Threshold:', 600, 1200, 800, 10)
pannel_setting_more.subheader('Person Track')
conf_track_plot_length = pannel_setting_more.slider('Track plot length', 10, 50, 25, 5)
conf_track_plot_size = pannel_setting_more.slider('Track plot size', 10, 40, 20, 1)
conf_person_occupy_rate = pannel_setting_more.slider('Person Size', 0.4, 0.8, 0.6, 0.05)
conf_super_resolve = pannel_setting_more.checkbox('Super Resolution', True)
pannel_setting_more.subheader('Global')
conf_frame_extraction = pannel_setting_more.slider('Frame Extraction', 0, 10, 0, 1)
conf_input_scale = pannel_setting_more.slider('Image Scale', 0.1, 1.0, 0.25, 0.05)
conf_multi_process = pannel_setting_more.selectbox('Multi Process', ['Off', '2', '4', '8'])
conf_multi_process = {
    'Off': -1,
    '2': 2,
    '4': 4,
    '8': 8
}[conf_multi_process]
conf_gpu_acceleration = pannel_setting_more.checkbox('GPU Acceleration', False)

## Main Page Construct
compo_status_bar = pannel_body.empty()
box_header_icon,_,_,_,_,_,_,_,_,_,_,_,_,_,_  = pannel_body.columns(15)
box_status = pannel_body.empty()
box_status.subheader('🔘 Class Over')
box_progress = pannel_body.progress(0)
box_header_icon.image(PATH_ICON_SMALL)
pannel_lecturer, pannel_slides, pannel_history = pannel_body.columns(3)

box_slide = pannel_lecturer.empty()
box_history = pannel_lecturer.container()
box_history.subheader('🗊 Slides Record')

box_slide_stroke, box_people = pannel_slides.columns(2)
box_origin = box_people.empty()
box_slide_stroke = box_slide_stroke.empty()
pannel_slides.subheader('🗈 Slide Note')
box_ocr_status = pannel_slides.empty()
box_slide_note = pannel_slides.empty()

box_people = pannel_history.empty()

pannel_history.subheader('📍 Track Path')
box_track = pannel_history.empty()
box_track = box_track.empty()

## Status Pannel Construct
pannel_runtime.subheader('FPS')
box_fps= pannel_runtime.empty()
pannel_runtime.subheader('Time Explain')
box_runtime = pannel_runtime.empty()

## Copyright Construct
pannel_copyright = pannel_setting.expander('©️ Copyright')
pannel_copyright.markdown('`X-Room ©️ Copyright 2023 Jiarui LI`')
pannel_copyright.subheader('Author')
pannel_copyright.markdown('👨‍🎓 Jiarui LI')
pannel_copyright.markdown('🏫 University of Nottingham Ningbo China')
pannel_copyright.markdown('📫 scyjl6@nottingham.edu.cn')

pannel_copyright.subheader('Project')
pannel_copyright.image(PATH_ICON)
pannel_copyright.markdown('**X-Room**')
pannel_copyright.markdown('Virtual classroom')
pannel_copyright.markdown('`COMP3052 UNNC` Computer Vision Project')
pannel_copyright.markdown('Supervised by Dr.Lu Zheng')


## Page Initialize (Put Placeholder)
box_people.image(PATH_PLACEHOLDER)
box_origin.image(PATH_PLACEHOLDER)
box_track.image(PATH_PLACEHOLDER)
box_slide.image(PATH_PLACEHOLDER)
box_slide_stroke.image(PATH_PLACEHOLDER)
box_slide_note.info('Board OCR Information')

############## Video Mode ##############
if control_start_play:
    compo_status_bar.info('⏳ Loading Model...')
    cap = cv2.VideoCapture(control_path_video)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    origin_fps = cap.get(cv2.CAP_PROP_FPS)
    counter = 0
    flow_model = load_flow_model(conf_background_path, conf_gpu_acceleration,
                             conf_ocr_refresh_thres, conf_lang_set,
                             conf_track_plot_length, conf_person_occupy_rate,
                             conf_multi_process, conf_input_scale, conf_super_resolve,
                             conf_track_plot_size, control_style_select)

    compo_status_bar.empty()
    box_progress.progress(0)
    current_frame = 0
    begin_time = time.time()
    while(cap.isOpened()):
        start_time = time.time()
        ret, frame = cap.read()
        if frame is None or not ret:
            break
        
        current_frame += 1
        current_time = time.time() - begin_time
        box_progress.progress(current_frame/frame_count, '🕙{:0>2d}:{:0>2d}:{:0>2d}'.format(int(current_time/360),
                                                                   int(current_time/60%60),
                                                                   int(current_time%60)))
        
        if counter < conf_frame_extraction:
            counter += 1
            continue
        else:
            counter = 0
            
        box_status.subheader('🔴 In Class')
        box_origin.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        flow = flow_model.forward(frame)
        board = flow.get('board')
        people = flow.get('people')
        
        box_slide.image(board['enhanced'][0][0])
        box_slide_stroke.image(board['stroke'][0][0])
        if board['ocr'][1] != -1:
            box_slide_note.table(board['ocr'][0][0][0])
            if board['ocr'][0][1][0]:
                if len(board['ocr'][0][0][0]) > 0:
                    pannel = box_history.expander(board['ocr'][0][0][0][0])
                    slide, note = pannel.tabs(['Slide', 'Note'])
                    slide.image(board['stroke'][0][0])
                    note.table(board['ocr'][0][0][0])
        else:
            box_slide_note.info('⏳ Note Refreshing...')
        
        for person in people['people'][0]:
            box_people.image(person)
        box_track.plotly_chart(people['path'][0], True)
        
        fps = 1 / (time.time() - start_time)
        
        box_fps.metric('Real Time', '{:.1f}'.format(fps), '{}'.format(int(fps-origin_fps)))
        
        box_runtime.write(flow_model.run_time())
    box_status.subheader('🔘 Class Over')
    box_people.image(PATH_PLACEHOLDER)
    box_origin.image(PATH_PLACEHOLDER)
    st.balloons()

############## Camera Mode ##############
if control_start_camera:
    cap = cv2.VideoCapture(0)
    origin_fps = cap.get(cv2.CAP_PROP_FPS)
    flow_model = load_flow_model(conf_background_path, conf_gpu_acceleration,
                             conf_ocr_refresh_thres, conf_lang_set,
                             conf_track_plot_length, conf_person_occupy_rate,
                             conf_multi_process, conf_input_scale, conf_super_resolve,
                             conf_track_plot_size)

    compo_status_bar.empty()
    box_progress.progress(0)
    current_frame = 0
    begin_time = time.time()
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None or not ret:
            break
        control_camera_preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        start_time = time.time()
        ret, frame = cap.read()
        if frame is None or not ret:
            break
        
        current_frame += 1
        current_time = time.time() - begin_time
        box_progress.progress(0, '🕙{:0>2d}:{:0>2d}:{:0>2d}'.format(int(current_time/360),
                                                                   int(current_time/60%60),
                                                                   int(current_time%60)))
            
        box_status.subheader('🔴 In Class')
        box_origin.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        flow = flow_model.forward(frame)
        board = flow.get('board')
        people = flow.get('people')
        
        box_slide.image(board['enhanced'][0][0])
        box_slide_stroke.image(board['stroke'][0][0])
        if board['ocr'][1] != -1:
            box_slide_note.table(board['ocr'][0][0][0])
            if board['ocr'][0][1][0]:
                if len(board['ocr'][0][0][0]) > 0:
                    pannel = box_history.expander(board['ocr'][0][0][0][0])
                    slide, note = pannel.tabs(['Slide', 'Note'])
                    slide.image(board['stroke'][0][0])
                    note.table(board['ocr'][0][0][0])
        else:
            box_slide_note.info('⏳ Note Refreshing...')
        
        for person in people['people'][0]:
            box_people.image(person)
        box_track.plotly_chart(people['path'][0], True)
        
        fps = 1 / (time.time() - start_time)
        box_fps.metric('Real Time', '{:.1f}'.format(fps), '{}'.format(int(fps-origin_fps)))
        
        box_runtime.write(flow_model.run_time())
        

