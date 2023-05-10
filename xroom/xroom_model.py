import cv2
import time
from utils import *
from config import *

def load_flow_model(background_path, gpu=False, ocr_refresh_thres=800,
                    ocr_lang=['en'], track_plot_length=25, person_occupy_rate=0.6,
                    multi_process=-1, input_scale=0.5, super_resolve=True,
                    track_plot_size=10):
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
                FlowModuleAsync(
                    OpticalCharRecognition(lang=ocr_lang, gpu=gpu, refresh_thres=ocr_refresh_thres)
                )
            ])
        }, multi_process=multi_process),
    ])
    
    if super_resolve:
        people_flow = FlowModuleList([
            ScaleImage(rate=input_scale),
            PeopleTrack(model=WEIGHT_YOLOV8SEG_PATH, gpu=gpu, max_people=1),
            FlowModuleBranch({
                'people': FlowModuleList([
                    SuperResolution(weights=WEIGHT_DRRN_PATH, residual_layers=9, gpu=gpu),
                    PersonBackgroundFuse(background=background_path, occupy_rate=person_occupy_rate),
                    BGR2RGB()
                ]),
                'path': TrackPlot(max_length=track_plot_length, plot_size=track_plot_size)
            })  
        ])
    else:
        people_flow = FlowModuleList([
            ScaleImage(rate=input_scale),
            PeopleTrack(model=WEIGHT_YOLOV8SEG_PATH, gpu=gpu, max_people=1),
            FlowModuleBranch({
                'people': FlowModuleList([
                    PersonBackgroundFuse(background=background_path, occupy_rate=person_occupy_rate),
                    BGR2RGB()
                ]),
                'path': TrackPlot(max_length=track_plot_length, plot_size=track_plot_size)
            })  
        ])

    flow_model = FlowModuleBranch({
        'people': people_flow,
        'board': board_flow
    }, multi_process=multi_process)
    
    return flow_model