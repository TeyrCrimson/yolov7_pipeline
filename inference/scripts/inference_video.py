import math
import argparse
from datetime import datetime
from importlib_resources import files
from pathlib import Path

import cv2

from yolov7.yolov7 import YOLOv7

def inference_video(args):
    yolov7 = YOLOv7(
        weights=files('yolov7').joinpath(args.yolov7_weights),
        cfg=files('yolov7').joinpath(args.yolov7_cfg),
        bgr=True,
        gpu_device=0,
        # model_image_size=640,
        model_image_size=640,
        max_batch_size=64,
        half=True,
        same_size=True,
        conf_thresh=0.25,
        trace=False,
        cudnn_benchmark=False,
    )

    for vid_path in Path(args.vid_folder).glob('*.mp4'):
        if not vid_path.is_file():
            raise AssertionError(f'{str(vid_path)} not found')

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_fp = output_dir / args.output_vid
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        print(out_fp)
        display_video = args.display_video



        vidcap = cv2.VideoCapture(str(vid_path))
        if not vidcap.isOpened():
            raise AssertionError(f'Cannot open video file {str(vid_path)}')

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fps = 25 if math.isinf(fps) else fps
        vid_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_track = cv2.VideoWriter(str(out_fp), cv2.VideoWriter_fourcc(*'MJPG'), fps, (vid_width, vid_height))

        if display_video:
            cv2.namedWindow('YOLOv7', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = vidcap.read()
            if not ret:
                break

            dets = yolov7.detect_get_box_in([frame], box_format='ltrb', classes=None)[0]
            if len(dets) > 0:
                print(dets)
            show_frame = frame.copy()
            for det in dets:
                ltrb, conf, clsname = det
                l, t, r, b = ltrb
                cv2.rectangle(show_frame, (l, t), (r, b), (255, 255, 0), 1)
                cv2.putText(show_frame, f'{clsname}:{conf:0.2f}', (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

            out_track.write(show_frame)

            if display_video:
                cv2.imshow('YOLOv7', show_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        if display_video:
            cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Live Video Inferencing using Yolov7 models')

parser.add_argument('--yolov7_weights', help='.pt file for the yolov7 model to point to', default='weights/cutpaste_w_cutpasteUS_w_allns_finetune_w_processed(ns)_1280_bs16_e102/reparam_state.pt')
parser.add_argument('--yolov7_cfg', help='.yaml file for the yolov7 model to point to', default='cfg/deploy/yolov7-tiny.yaml')
parser.add_argument('--vid_folder', help='location of video source. Can be either filepath or URL', default='inference/vids')
parser.add_argument('--output_dir', help='directory to save the inferenced video', default='inference')
parser.add_argument('--output_vid', help='relative path of the final save location w.r.t. output_dir', default=datetime.now().strftime("%m-%d-%Y/%H-%M-%S") + '.avi')
parser.add_argument('-d', '--display_video', help='Indicate if you want the video to be shown when inferencing', action='store_true')

args = parser.parse_args()
inference_video(args)
