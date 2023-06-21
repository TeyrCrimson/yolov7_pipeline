import argparse
from pathlib import Path
import numpy as np
import cv2

from yolov7.yolov7 import YOLOv7

import pyrealsense2 as rs

def main(weights, cfg, savepath, fps, vid_width, vid_height):
  
  # Check inputs
  if savepath and savepath.is_path_exists_or_creatable():
    savepath.parent.mkdir(parents=True, exist_ok=True)
    out_filepath = Path(savepath)
    print(f"savepath indicated. saving output as {str(out_filepath)}")
  else:
    output_dir = Path('inference')
    output_dir.mkdir(parents=True, exist_ok=True)
    out_filepath = output_dir / "output_inference.mp4"
    print(f"WARNING: savepath not indicated or savepath not a valid filepath.\nsaving output as {str(out_filepath)}")

  if not weights.is_file():
    raise FileNotFoundError(f"unable to find weights file: {weights}")

  if not cfg.is_file():
    raise FileNotFoundError(f"unable to find cfg file: {cfg}")

  # Initialise inference model
  yolov7 = YOLOv7(
    weights=weights,
    cfg=cfg,
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

  # Configure depth and color streams
  pipeline = rs.pipeline()
  config = rs.config()

  # Get device product line for setting a supporting resolution
  pipeline_wrapper = rs.pipeline_wrapper(pipeline)
  pipeline_profile = config.resolve(pipeline_wrapper)
  device = pipeline_profile.get_device()

  found_rgb = False
  for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
      found_rgb = True
      print(f"using device {s}")
      break
  if not found_rgb:
    print("The demo requires a Depth camera with Color sensor")
    exit(0)

  print(f"stream configs: {vid_width}x{vid_height} with {fps} fps")
  config.enable_stream(rs.stream.color, vid_width, vid_height, rs.format.yuyv, fps)

  ann_writer = cv2.VideoWriter(str(out_filepath), cv2.VideoWriter_fourcc(*'h264'), float(fps), (vid_width, vid_height))

  # Start streaming
  pipeline.start(config)
  print(f"starting stream, press q to exit")

  try:
    while True:
      # Wait for coherent frames
      frames = pipeline.wait_for_frames()
      color_frame = frames.get_color_frame()
      if not color_frame:
        continue

      # frame = np.asanyarray(color_frame.get_data())
      frame = yuyv_to_bgr(color_frame)

      dets = yolov7.detect_get_box_in([frame], box_format='ltrb', classes=None)[0]

      show_frame = frame.copy()
      for det in dets:
        ltrb, conf, clsname = det
        l, t, r, b = ltrb
        cv2.rectangle(show_frame, (l, t), (r, b), (255, 255, 0), 1)
        cv2.putText(show_frame, f'{clsname}:{conf:0.2f}', (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

      ann_writer.write(show_frame)
      
      # Show images
      cv2.namedWindow('RealSense YOLOv7', cv2.WINDOW_AUTOSIZE)
      cv2.imshow('RealSense YOLOv7', show_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    cv2.destroyAllWindows()
    ann_writer.release()

  finally:
    # Stop streaming
    pipeline.stop()

def yuyv_to_bgr(color_frame):
  h = color_frame.get_height()
  w = color_frame.get_width()
  y = np.frombuffer(color_frame.get_data(), dtype=np.uint8)[0::2].reshape(h, w)
  uv = np.frombuffer(color_frame.get_data(), dtype=np.uint8)[1::2].reshape(h, w)
  yuv = np.zeros((h, w, 2), 'uint8')
  yuv[:,:,0] = y
  yuv[:,:,1] = uv
  bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
  return bgr

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--weights', type=Path, required=True, help='path to weights')
  parser.add_argument('-c', '--cfg', type=Path, required=True, help='path to config file')
  parser.add_argument('--savepath', type=Path, default=None, help='desired filepath of output video')
  parser.add_argument('--width', type=int, default=1280, help='desired width of video')
  parser.add_argument('--height', type=int, default=720, help='desired height of video')
  parser.add_argument('--fps', type=int, default=30, help='desired fps of video')
  opt = parser.parse_args()
  main(opt.weights, opt.cfg, opt.savepath, opt.fps, opt.width, opt.height)
