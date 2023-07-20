import math
import random
import argparse
from importlib_resources import files
from pathlib import Path

import cv2

STANDARD_SIZE = (1280, 720)

def createEmptyFile(filepath):
    f = open(filepath, 'a')
    f.close()


def vid_routine_ns(path):
    for vid_path in Path(path).glob('*.mp4'):
        # vid_path = Path('testvideo.mp4')
        if not vid_path.is_file():
            raise AssertionError(f'{str(vid_path)} not found')

        output_dir = Path('inference')
        output_dir.mkdir(parents=True, exist_ok=True)
        # out_fp = output_dir / f'{vid_path.stem}_inference.avi'
        display_video = False

        vidcap = cv2.VideoCapture(str(vid_path))
        print(vidcap.isOpened(), vidcap.get(cv2.CAP_PROP_FPS))
        if not vidcap.isOpened():
            raise AssertionError(f'Cannot open video file {str(vid_path)}')

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fps = 25 if math.isinf(fps) else fps
        vid_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        viewMode = None
        if vid_height > vid_width:
            print('[INFO] Portrait video detected')
            viewMode == "portrait"

        frame_no = 0
        while True:
            ret, frame = vidcap.read()
            # if viewMode == "portrait": 
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            if not ret:
                break
            show_frame = frame.copy()
            margin_h = int(vid_height * random.randrange(100, 200)/1000)
            show_frame = show_frame[margin_h:-margin_h, 0:vid_width]
            # ratios = [standard/original for standard, original in zip(STANDARD_SIZE, tuple(show_frame.shape))]
            resized = cv2.resize(show_frame, STANDARD_SIZE, interpolation = cv2.INTER_AREA)
            Path('./processed/images').mkdir(parents=True, exist_ok=True)
            Path('./processed/labels').mkdir(parents=True, exist_ok=True)
            iname = "./processed/images/{}_frame{}.jpg".format(vid_path.stem, frame_no)
            lname = "./processed/labels/{}_frame{}.txt".format(vid_path.stem, frame_no)
            cv2.imwrite(iname, resized)
            createEmptyFile(lname)
            frame_no += 1

            if display_video:
                cv2.imshow('YOLOv7', show_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        if display_video:
            cv2.destroyAllWindows()

def pic_routine_ns(path):
    pic_files = [p for p in Path(path).glob('*') if p.suffix in ['.jpg', '.jpeg', '.png', '']]
    print(pic_files)
    for pic_path in pic_files:
        img = cv2.imread(str(pic_path))
        resized = cv2.resize(img, STANDARD_SIZE, interpolation = cv2.INTER_AREA)
        Path('./processed/images').mkdir(parents=True, exist_ok=True)
        Path('./processed/labels').mkdir(parents=True, exist_ok=True)
        iname = str(Path('./processed/images') / pic_path.name)
        lname = str(Path('./processed/labels') / Path(pic_path.stem + '.txt'))
        try:
            cv2.imwrite(iname, resized)
        except Exception as e:
            print('{}, defaulting to .jpg')
            cv2.imwrite(iname + '.jpg', resized)
        createEmptyFile(lname)

parser = argparse.ArgumentParser(description='Data Preparer')

parser.add_argument('path', help='Directory to process videos or images')
parser.add_argument('-v', '--vid', help='indicate if the processed item is a directory of videos', action='store_true')
parser.add_argument('-i', '--img', help='indicate if the processed item is a directory of images', action='store_true')
parser.add_argument('--ns', help='indicate if the processed directory is for negative samples', action='store_true')

args = parser.parse_args()
assert args.vid or args.img, 'Please indicate if the directory contains videos or images by indicating -v or -i' 
if args.ns:
    vid_routine_ns(args.path) if args.vid else pic_routine_ns(args.path)
        
