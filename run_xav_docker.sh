WORKSPACE=/mnt/rootfs/realsense

V4L_SYMLINK=/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_455_Intel_R__RealSense_TM__Depth_Camera_455_120623060537-video-index

xhost +local:docker
docker run -it --rm --net=host \
	--runtime nvidia \
	--platform=linux/arm64 \
	--privileged \
	--device $V4L_SYMLINK'0' \
	--device $V4L_SYMLINK'1' \
	--device $V4L_SYMLINK'2' \
	--device $V4L_SYMLINK'3' \
	--device $V4L_SYMLINK'4' \
	--device $V4L_SYMLINK'5' \
	-w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	--ipc=host \
	-v $HOME/.Xauthority:/root/.Xauthority:rw \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=unix$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	yolov7-realsense-arm
