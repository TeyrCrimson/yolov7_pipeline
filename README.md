# YOLOv7 Package with RealSense

## Adapted/Forked from WongKinYiu's Repository

- Repository Link: https://github.com/WongKinYiu/yolov7
- Last "merge" date: 7th Sept 2022

### Updates to the Repository

The following changes have been made to the original repository:

- YOLOv7 Modification:
    - YOLOv7 has been modified to function as a **package** solely for inference purposes.
    - Modifications were made to the `inference` branch to enable compatibility with **RealSense cameras**.

- ARM Processor Support:
    - This repository is **compatible with ARM processors**, allowing users to run the inference process on these devices.

## Table of Contents
- [1. Running on NVIDIA Jetson Xavier](#1-running-on-nvidia-jetson-xavier)
  - [Setup on a New Xavier](#setup-on-a-new-xavier)
    - [Requirements](#requirements)
    - [Setup Instructions](#setup-instructions)
  - [Running a Live YOLOv7 Inference Script with a RealSense Camera](#running-a-live-yolov7-inference-script-with-a-realsense-camera)
- [2. Running on AMD Processors](#2-running-on-amd-processors)
  - [Quick Start](#quick-start)
- [Official YOLOv7](#official-yolov7)

## 1. Running on NVIDIA Jetson Xavier

### Setup on a New Xavier

This program should work on ARM processors in general but has only been tested on the NVIDIA Jetson Xavier.

#### Requirements
- Ubuntu OS
- External hard disk with Linux File System (eg. ext4) for extra storage 
- RealSense Camera

#### Setup Instructions

Please execute the following instructions in order when setting up a new NVIDIA Jetson Xavier.

1. **Flash Ubuntu**

    TODO

1. **Date & Time Adustment (if applicable)**

    In the event that the date and time on the NVIDIA Jetson Xavier device are inaccurate, please make sure to adjust them each time the device is rebooted. Otherwise, there may be complications during program installation and execution.

1. **External Disk Automount Configuration**

    Automount the SSD/HDD/SD card onto `/mnt/rootfs`. Ensure that the disk is of Linux File System type.

    1. Identify the **UUID** and **file system type** of your drive by executing the following command:
        ```bash
        sudo blkid
        ```

    1. Create a mount point for your drive under the `/mnt` directory. In this example, we will use `/mnt/rootfs`.
        ```bash
        sudo mkdir /mnt/rootfs
        ```

    1. Append the following line to the `/etc/fstab` file using your preferred text editor:
        ```
        UUID=<uuid-of-your-drive>  <mount-point>  <file-system-type>  <mount-option>  <dump>  <pass>
        ```
        For example,
        ```bash
        UUID=eb67c479-962f-4bcc-b3fe-cefaf908f01e  /mnt/rootfs  ext4  defaults  0  2
        ```

    1. Verify the automount configuration by executing the following command:
        ```bash
        sudo mount -a
        ```

    For additional information and details, please refer to the following [link](https://www.linuxbabe.com/desktop-linux/how-to-automount-file-systems-on-linux).

1. **Docker Installation**

    Before proceeding with the installation of Docker, ensure that no other versions of Docker are currently installed. If any are found, please **uninstall** them prior to continuing. 
    1. Download and install Docker for ARM using the following commands:
        ```bash
        sudo apt-get update
        sudo apt-get upgrade
        curl -fsSL test.docker.com -o get-docker.sh && sh get-docker.sh
        sudo usermod -aG docker $USER 
        ```

    1. Log out of the system and then log back in to apply the group membership changes. Verify the successful installation of Docker by running the following command:
        ```bash
        docker run hello-world 
        ```
    For more detailed information and instructions, please refer to the following [link](https://www.docker.com/blog/getting-started-with-docker-for-arm-on-linux/)

1. **NVIDIA Container Toolkit Installation (for GPU Usage)**

    1. To install the NVIDIA Container Toolkit on the NVIDIA Jetson Xavier, execute the following commands:
        ```bash
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        sudo apt install -y nvidia-docker2
        sudo systemctl daemon-reload
        sudo systemctl restart docker
        ```
    
    Note: When prompted to allow changes to `/etc/docker/daemon.json` during installation, accept the changes.
    For more detailed information and instructions, please refer to the following [link](https://dev.to/caelinsutch/running-docker-containers-for-the-nvidia-jetson-nano-5a06)

1. **Relocating Docker Storage (if required) - Insufficient Storage Capacity**

    Due to the limited storage capacity of the NVIDIA Jetson Xavier (16GB), it may become necessary to relocate Docker's storage location to an external hard disk. Follow the steps below:

    1. Stop the Docker daemon
        ```bash
        sudo service docker stop
        ```
    1. Open the `/etc/docker/daemon.json` file using a text editor and add the following JSON configuration:
        ```json
        {
            "data-root": "/path/to/your/docker"
        }
        ```
        Replace `/path/to/your/docker` with the desired path on the external hard disk.

    1. Create a new directory on the external hard disk to store Docker's data and transfer the data over. For instance, we will use the directory `/mnt/rootfs/docker`.
        ```bash
        sudo mkdir /mnt/rootfs/docker
        sudo rsync -aP /var/lib/docker/ /mnt/rootfs/docker
        sudo mv /var/lib/docker /var/lib/docker.old
        sudo service docker start
        ```

    For more detailed information and instructions, please refer to the following [link](https://www.guguweb.com/2019/02/07/how-to-move-docker-data-directory-to-another-location-on-ubuntu/).

1. **Obtaining the Symlink for the Video Device**

    1. To identify the symlink of your video device, execute the following command:
        ```bash
        sudo udevadm info --query=all --name=/dev/video1
        ```
    Take note of the v4l `by-id` symlink information provided. 
    Update the video device symlink accordingly in `run_docker.sh` to ensure proper device mapping and usage.

### Running a Live YOLOv7 Inference Script with a RealSense Camera

Follow the steps below to run a live YOLOv7 inference script using a RealSense camera:

1. Clone this repository and switch to the `realsense` branch
1. Build the Docker image by executing the following command:
    ```
    docker build -f Dockerfile.xav -t yolov7 .
    ```
1. Create and enter the Docker container using the provided script:
    ```
    bash run_xav_docker.sh
    ```
1. Import the wrapper classes for YOLOv7 and pyrealsense2 for inference, or use the provided scripts for inference with the RealSense camera:
    1. Import the classes manually in your script:
        ```
        import pyrealsense2 as rs
        from yolov7.yolov7 import YOLOv7
        ```
    1. Alternatively, you can directly use the provided scripts `xavier/inference_rs.py` or `xavier/run_inference_rs.sh` for inference:
    - Use the following command to run the inference script directly:
        ```
        python xavier/inference_rs.py -w /path/to/weights.file -c /path/to/deploy_cfg.file --inference-folder /path/to/inference_folder --raw-video-folder /path/to/raw_video_folder --width desired_vid_width --height desired_vid_height --fps desired_fps
        ```
    - Alternatively, navigate to the `xavier` directory and execute the provided script:
        ```
        cd xavier
        bash run_inference_rs.sh
        ```

Notes: 
- If the RealSense camera is not functioning, ensure that it is connected to a USB 3.0 port rather than a USB 2.0 port.
- If you encounter any issues, consider checking the device configuration in the `./run_docker` script if you are not using the RealSense D455 camera.
- For debugging purposes, you can download and use the [realsense-viewer application](https://dev.intelrealsense.com/docs/nvidia-jetson-tx2-installation).

## 2. Running on AMD Processors

### Quick Start

1. Clone this repository and switch to the `realsense` branch
1. Build the Docker image by executing the following command:
    ```
    docker build -f Dockerfile.amd -t yolov7 .
    ```
1. Create and enter the Docker container using the provided script:
    ```
    bash run_amd_docker.sh
    ```

1. Import the wrapper classes for YOLOv7 and pyrealsense2 for inference, or use the provided scripts for inference with the RealSense camera:
    1. Import the classes manually in your script:
        ```
        import pyrealsense2 as rs
        from yolov7.yolov7 import YOLOv7
        ```
    1. Alternatively, you can directly use the provided scripts `scripts/inference_rs.py` or `scripts/run_inference_rs.sh` for inference:
    - Use the following command to run the inference script directly:
        ```
        python scripts/inference_rs.py -w /path/to/weights -c /path/to/deploy/cfg --savepath /path/to/savepath --width desired_vid_width --height desired_vid_height --fps desired_fps
        ```
    - Alternatively, navigate to the `scripts` directory and execute the provided script:
        ```
        cd scripts
        bash run_inference_rs.sh
        ```

# Official YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>

## Performance

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
