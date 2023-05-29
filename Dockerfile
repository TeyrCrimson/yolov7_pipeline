# docker build -t yolov7-inference-rs .
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV cwd="/home/"
WORKDIR $cwd

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt -y update

RUN apt-get install --no-install-recommends -y \
    software-properties-common \
    build-essential \
    git curl libgl1-mesa-glx libgtk2.0-dev pkg-config libssl-dev libusb-1.0-0-dev libgtk-3-dev \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libcurl4-openssl-dev

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

RUN python3 -m pip install --upgrade pip setuptools

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# YOLOv7
RUN git clone -b inference https://github.com/yhsmiley/yolov7.git && cd yolov7 && python3 -m pip install --no-cache-dir .

# RealSense
# https://github.com/IntelRealSense/librealsense/issues/6964
# https://github.com/IntelRealSense/librealsense/issues/6980#issuecomment-666858977
# https://github.com/IntelRealSense/librealsense/issues/7722

RUN wget http://www.cmake.org/files/v3.13/cmake-3.13.0.tar.gz \
    && tar xpvf cmake-3.13.0.tar.gz cmake-3.13.0/ \
    && cd cmake-3.13.0/ \
    && ./bootstrap --system-curl \
    && make -j6 \
    && echo 'export PATH=/home/nvidia/cmake-3.13.0/bin/:$PATH' >> ~/.bashrc \
    && source ~/.bashrc

RUN git clone https://github.com/IntelRealSense/librealsense.git \
    && cd ./librealsense \
    && echo 'set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../include")' | cat - /home/librealsense/CMake/install_config.cmake > temp \
    && mv temp /home/librealsense/CMake/install_config.cmake \
    && rm -rf build \
    && mkdir build \
    && cd build \
    && cmake \
    # -DCMAKE_BUILD_TYPE=debug \
    # -DCMAKE_INSTALL_PREFIX=/home/librealsense/install_host \
    # -DPYTHON_INSTALL_DIR=/usr/lib/python3.8/site-packages/pyrealsense2 \
    # -DBUILD_EXAMPLES=true \
    # -DFORCE_LIBUVC=true \
    -DBUILD_WITH_CUDA=true \
    -DBUILD_PYTHON_BINDINGS=bool:true \
    # -DPYBIND11_INSTALL=ON \
    # -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    # -DPYTHON_EXECUTABLE=$(which python3) \
    ../ \
    && make uninstall \
    && make clean \
    && make -j4 \
    && make install

RUN cp /home/librealsense/wrappers/python/pyrealsense2/__init__.py /usr/lib/python3.8/site-packages/pyrealsense2/

ENV PYTHONPATH "${PYTHONPATH}:/usr/lib/python3.8/site-packages"
