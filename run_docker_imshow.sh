xhost +local:docker
docker run -it --rm --gpus all  --shm-size 4GiB \
  --net host \
  --ipc host \
  -v $WORKSPACE:$WORKSPACE \
  -v $HOME/.Xauthority:/root/.Xauthority:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${PWD}:/testing \
  -e DISPLAY=unix$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  yolov7_inference:ver1.4
xhost -local:docker