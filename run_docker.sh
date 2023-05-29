WORKSPACE=/media/data/yolov7
DATA=/media/data/datasets

xhost +local:docker
docker run -it --rm \
	--platform=linux/amd64 \
	--gpus all \
	--device /dev/video1 \
	--device /dev/video2 \
	--device /dev/video3 \
	--device /dev/video4 \
	--device /dev/video5 \
  -w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	-v $DATA:$DATA \
	--ipc=host \
  -v $HOME/.Xauthority:/root/.Xauthority:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=unix$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
	yolov7-inference-rs
