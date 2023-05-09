import streamlit as st
import cv2
import time
from utils import *

WEIGHT_YOLOV8SEG_PATH = './weights/yolov8n-seg.pt'
WEIGHT_DRRN_PATH = './weights/drrn_B1U9.pth'
background_packs = {
    'Sky': './backgrounds/sky.jpg',
    'Forest': './backgrounds/forest.jpg',
    'Mountain': './backgrounds/mountain.jpg',
    'Night': './backgrounds/night.jpg',
    'Valley': './backgrounds/valley.jpg'
}
language_map = {
    'English': 'en',
    'Simplified Chinese': 'ch_sim',
    'Traditional Chinese': 'ch_tra',
    'Spanish': 'es',
    'French': 'fr',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Russian': 'ru',
    'Italian': 'it'
}
PATH_PLACEHOLDER = './image/placeholder.png'
PATH_ICON = './image/icon-long.png'
PATH_ICON_SMALL = './image/icon.png'
PATH_DEFAULT_VIDEO = '../video/sample-0.mp4'

@st.cache_resource
def load_people_track_model(gpu=False):
    return PeopleTrack(model=WEIGHT_YOLOV8SEG_PATH, gpu=gpu, max_people=1)

@st.cache_resource
def load_optical_char_recog_model(lang=['en'], gpu=False, refresh_thres=800):
    return FlowModuleAsync(
        OpticalCharRecognition(lang=lang, gpu=gpu, refresh_thres=refresh_thres)
    )

@st.cache_resource
def load_super_resolve_model(gpu=False):
    return SuperResolution(weights=WEIGHT_DRRN_PATH, residual_layers=9, gpu=gpu)

@st.cache_resource
def load_flow_model(background_path, gpu=False, ocr_refresh_thres=800,
                    ocr_lang=['en'], track_plot_length=25, person_occupy_rate=0.6,
                    multi_process=-1, input_scale=0.5, super_resolve=True):
    board_flow = FlowModuleList([
        BoardTracker(max_board=1, transition=0.9, drop_thres=0.97),
        FlowModuleBranch({
            'enhanced': FlowModuleList([
                ImageEnhance(sharpen=0.1),
                BGR2RGB()
            ]),
            'stroke': StrokeImage(strenth=6),
            'ocr': FlowModuleList([
                StrokeImage(strenth=6),
                load_optical_char_recog_model(lang=ocr_lang, gpu=gpu, refresh_thres=ocr_refresh_thres),
            ])
        }, multi_process=multi_process),
    ])
    
    if super_resolve:
        people_flow = FlowModuleList([
            ScaleImage(rate=input_scale),
            load_people_track_model(gpu=gpu),
            FlowModuleBranch({
                'people': FlowModuleList([
                    load_super_resolve_model(gpu=gpu),
                    PersonBackgroundFuse(background=background_path, occupy_rate=person_occupy_rate),
                    BGR2RGB()
                ]),
                'path': TrackPlot(max_length=track_plot_length)
            })  
        ])
    else:
        people_flow = FlowModuleList([
            ScaleImage(rate=input_scale),
            load_people_track_model(gpu=gpu),
            FlowModuleBranch({
                'people': FlowModuleList([
                    PersonBackgroundFuse(background=background_path, occupy_rate=person_occupy_rate),
                    BGR2RGB()
                ]),
                'path': TrackPlot(max_length=track_plot_length)
            })  
        ])

    flow_model = FlowModuleBranch({
        'people': people_flow,
        'board': board_flow
    }, multi_process=multi_process)
    
    return flow_model


st.set_page_config(page_title='XRoom', layout='wide', page_icon=PATH_ICON_SMALL)
hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

pannel_control = st.sidebar.container()
pannel_setting = st.sidebar.container()
pannel_body = st

icon,_ = pannel_control.columns(2)
icon.image(PATH_ICON)
control_path_video = pannel_control.text_input('🎞️ Video Path:', PATH_DEFAULT_VIDEO)
control_start_play = pannel_control.button(' ▶️ Play ')


pannel_setting.subheader('🛠️ Settings')
control_background_select = pannel_setting.selectbox(
    'Background',
    list(background_packs.keys())
)

conf_background_path = background_packs[control_background_select]
pannel_setting.image(conf_background_path)
pannel_setting_more = pannel_setting.expander('⚙️ Advanced Settings')
pannel_runtime = pannel_setting.expander('⏳ Run Time')
pannel_setting_more.subheader('Board')
conf_lang_set = pannel_setting_more.multiselect('OCR Languages', list(language_map.keys()), ['English'])
conf_lang_set = [language_map.get(item) for item in conf_lang_set]
conf_ocr_refresh_thres = pannel_setting_more.slider('OCR Refresh Threshold:', 600, 1200, 800, 10)
pannel_setting_more.subheader('Person Track')
conf_track_plot_length = pannel_setting_more.slider('Track plot length', 10, 50, 25, 5)
conf_person_occupy_rate = pannel_setting_more.slider('Person Size', 0.4, 0.8, 0.6, 0.05)
conf_super_resolve = pannel_setting_more.checkbox('Super Resolution', True)
pannel_setting_more.subheader('Global')
conf_frame_extraction = pannel_setting_more.slider('Frame Extraction', 0, 16, 5, 1)
conf_input_scale = pannel_setting_more.slider('Image Scale', 0.1, 1.0, 0.25, 0.05)
conf_multi_process = pannel_setting_more.selectbox('Multi Process', ['Off', '2', '4', '8'])
conf_multi_process = {
    'Off': -1,
    '2': 2,
    '4': 4,
    '8': 8
}[conf_multi_process]
conf_gpu_acceleration = pannel_setting_more.checkbox('GPU Acceleration', False)


compo_status_bar = pannel_body.empty()
box_header_icon,_,_,_,_,_,_,_,_,_,_,_,_,_,_  = pannel_body.columns(15)
box_status = pannel_body.empty()
box_status.subheader('⏸️ Class Over')
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

pannel_runtime.subheader('FPS')
box_fps= pannel_runtime.empty()
pannel_runtime.subheader('Time Explain')
box_runtime = pannel_runtime.empty()


box_people.image(PATH_PLACEHOLDER)
box_origin.image(PATH_PLACEHOLDER)
box_track.image(PATH_PLACEHOLDER)
box_slide.image(PATH_PLACEHOLDER)
box_slide_stroke.image(PATH_PLACEHOLDER)
box_slide_note.info('Board OCR Information')

if control_start_play:
    compo_status_bar.info('⏳ Loading Model...')
    cap = cv2.VideoCapture(control_path_video)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    origin_fps = cap.get(cv2.CAP_PROP_FPS)
    counter = 0
    flow_model = load_flow_model(conf_background_path, conf_gpu_acceleration,
                             conf_ocr_refresh_thres, conf_lang_set,
                             conf_track_plot_length, conf_person_occupy_rate,
                             conf_multi_process, conf_input_scale, conf_super_resolve)

    compo_status_bar.empty()
    box_progress.progress(0)
    current_frame = 0
    while(cap.isOpened()):
        start_time = time.time()
        ret, frame = cap.read()
        if frame is None or not ret:
            break
        
        current_frame += 1
        box_progress.progress(current_frame/frame_count)
        
        if counter < conf_frame_extraction:
            counter += 1
            continue
        else:
            counter = 0
            
        box_status.subheader('▶️ In Class')
        box_origin.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        flow = flow_model.forward(frame)
        board = flow.get('board')
        people = flow.get('people')
        
        box_slide.image(board['enhanced'][0][0])
        box_slide_stroke.image(board['stroke'][0][0])
        if board['ocr'][1] != -1:
            box_slide_note.table(board['ocr'][0][0][0])
            if board['ocr'][0][1][0]:
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
        box_fps.markdown('**Real Time(Origin):** {:.2f}({:.2f})'.format(fps, origin_fps))
        
        box_runtime.write(flow_model.run_time())
    box_status.subheader('⏸️ Class Over')
    box_people.image(PATH_PLACEHOLDER)
    box_origin.image(PATH_PLACEHOLDER)
    st.balloons()