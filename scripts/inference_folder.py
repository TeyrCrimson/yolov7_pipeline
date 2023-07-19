import argparse
from datetime import datetime
from importlib_resources import files
from pathlib import Path

import cv2

from yolov7.yolov7 import YOLOv7

def inference_folder(args):
    src_folder = args.source_dir
    output_folder = args.output_dir
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    yolov7 = YOLOv7(
        weights=files('yolov7').joinpath(args.yolov7_weights),
        cfg=files('yolov7').joinpath(args.yolov7_cfg),
        bgr=True,
        gpu_device=0,
        model_image_size=640,
        max_batch_size=64,
        half=True,
        same_size=True,
        conf_thresh=0.25,
        trace=False,
        cudnn_benchmark=False,
    )

    all_imgpaths = [imgpath for imgpath in Path(src_folder).rglob("*.jpg")]
    all_imgs = [cv2.imread(str(imgpath)) for imgpath in all_imgpaths]

    all_dets = yolov7.detect_get_box_in(all_imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)
    try:
        for idx, dets in enumerate(all_dets):
            draw_frame = all_imgs[idx].copy()
            print(f'img {all_imgpaths[idx].name}: {len(dets)} detections')
            for det in dets:
                bb, score, class_ = det
                l,t,r,b = bb
                cv2.rectangle(draw_frame, (l,t), (r,b), (255,255,0), 1)
                cv2.putText(draw_frame, class_, (l, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0))

            output_path = Path(output_folder) / f'{all_imgpaths[idx].stem}_det.jpg'
            cv2.imwrite(str(output_path), draw_frame)
    except Exception as e:
        print('[ERROR]: {e}')

parser = argparse.ArgumentParser(description='Live Video Inferencing using Yolov7 models')

parser.add_argument('--yolov7_weights', help='.pt file for the yolov7 model to point to', default='weights/cutpaste_w_cutpasteUS_w_allns_finetune_w_processed(ns)_1280_bs16_e102/reparam_state.pt')
parser.add_argument('--yolov7_cfg', help='.yaml file for the yolov7 model to point to', default='cfg/deploy/yolov7-tiny.yaml')
parser.add_argument('--source_dir', help='location of video source. Can be either filepath or URL', default='inference/images')
parser.add_argument('--output_dir', help='directory to save the inferenced video', default='inference/results')

args = parser.parse_args()
inference_folder(args)
