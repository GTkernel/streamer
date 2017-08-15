## Installation Instructions

### 1. Dependencies

#### MacOS

```sh
brew install cmake glog glib gstreamer gst-plugins-base \
	gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-ffmpeg \
	boost opencv jemalloc zmq zmqpp eigen
```

#### Ubuntu

First [install OpenCV 3.x](http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html).

```sh
sudo apt update
sudo apt install -y cmake libglib2.0-dev libgoogle-glog-dev \
    libboost-all-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-libav libjemalloc-dev \
    libzmq3-dev libeigen3-dev
```

### 2. Compile and run

Please see the main [README](README.md) for a Quick Start guide on how to
compile and run Streamer.

### 3. Additional compilation options

* Control the backend
    - Use CPU: `-DBACKEND=cpu`
    - Use CUDA device: `-DBACKEND=cuda`
    - Use OpenCL device: `-DBACKEND=opencl`

* Add deep learning frameworks
    - Use Caffe: `-DUSE_CAFFE=yes -DCAFFE_HOME=/path/to/caffe/distribute`
    - Use TensorFlow:
    `-DUSE_TENSORFLOW=yes -DTENSORFLOW_HOME=/path/to/tensorflow`

* Enable additional camera types
    - Build with PtGray SDK for their GigE cameras: `-DUSE_PTGRAY=yes`
    - Build with Vimba SDK for Allied Vision cameras:
    `-DUSE_VIMBA=yes -DVIMBA_HOME=/path/to/Vimba_2_1`

* Configure RPC support. Note that Streamer already includes ZeroMQ support by
default, which can be used for publishing and subscribing to streams. This
option adds gRPC as an alternative way to connect Streamer instances running on
different machines.
    - Build with gRPC support: `-DUSE_RPC=yes`

### 4. Encoders and decoders

The default configuration tries to pick a good GStreamer encoder and decoder for
the given platform (i.e., prefer hardware over software). To specify other
decoders and encoders available on the system, edit the `config/config.toml` file.

Running the `gst-inspect-1.0` command (found in the `gstreamer1.0-tools` package
on Ubuntu) will list the available encoders and decoders. The list will vary
depending on which GStreamer plugins are installed.
