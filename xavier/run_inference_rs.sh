WEIGHTS='/mnt/rootfs/realsense/yolov7/weights/best.pt'
CFG='/mnt/rootfs/realsense/yolov7/cfg/deploy/yolov7-tiny.yaml'
INFERENCE_FOLDER='/mnt/rootfs/realsense/output/inference_videos'
RAW_VIDEO_FOLDER='/mnt/rootfs/realsense/output/raw_videos'

WIDTH=1280
HEIGHT=720
FPS=5

python3 inference_rs.py \
  -w $WEIGHTS \
  -c $CFG \
  --bgr \
  --gpu_device 0 \
  --model_image_size 1280 \
  --max_batch_size 64 \
  --conf_thresh 0.25 \
  --nms_thresh 0.45 \
  --inference-folder $INFERENCE_FOLDER \
  --raw-video-folder $RAW_VIDEO_FOLDER \
  --width $WIDTH \
  --height $HEIGHT \
  --fps $FPS
