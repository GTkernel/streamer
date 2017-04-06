## streamer

[![GitHub license](https://img.shields.io/badge/license-apache-green.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://travis-ci.com/viscloud/streamer.svg?token=Khh4txHVr3EZjYCRAdze&branch=master)](https://travis-ci.com/viscloud/streamer)

Streamer is a system built for real-time video ingestion and analytics of live camera streams, with a special focus on running state-of-the-art deep learning based vision algorithms. The goal of streamer is twofold:

* A near camera system that provides computation resources and programming models for real-time, low-latency tasks.
* A multi-camera infrastructure that provides camera controlling, sensor and application data storage and retrieval, tasks (potentially user-defined) planning and execution across a network of cameras.

**Current status**

You can run an end-to-end ingestion and analytics pipeline on streamer with multiple camera input. An example pipeline: 

* Take frames off cameras from RTSP endpoint
* Decode the stream with hardware h.264 decoder
* Transform, scale and normalize the frames into BGR image that could feed directly into a neural network
* Run an imagenet network (GoogleNet, AlexNet)
* Collect the classification results as well as the original video data, and store them locally on disk. Videos can be re-encoded to save disk space.



For DNN frameworks, streamer currently supports Caffe, Caffe FP16 (NVIDIA's fork), Caffe OpenCL, Caffe-MKL, MXNet and TensorRT (former GIE from NVIDIA). The support of tensorflow is on the way.

## Getting Started

### Dependencies

Note: Here I will only include minimal requirements to successfully build streamer without additional dependencies. You may want to install various deep learning frameworks (Caffe, MXNet, etc.). You can refer to their documentation on how to install them on your platform. 

#### 1. Mac


With homebrew:

```
brew install cmake glog glib gstreamer gst-plugins-base \
	gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-ffmpeg \
	boost opencv jemalloc zmq eigen
```

#### 2. Linux x86 (Ubuntu)

Ubuntu 16.04

```
# install OpenCV from http://opencv.org/
sudo apt-get update
sudo apt-get install -y cmake libglib2.0-dev libgoogle-glog-dev \
    libboost-all-dev libopencv-dev gstreamer1.0 libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev libjemalloc-dev libzmq3-dev libeigen3-dev
```

For Ubuntu <= 14.04, also need to install cmake 3.

```
sudo add-apt-repository ppa:george-edison55/cmake-3.x
sudo apt-get update
# When cmake is not installed
sudo apt-get install cmake
# When cmake is already installed
sudo apt-get upgrade
```

If the above does not work, you can [compile cmake 3 from source](http://askubuntu.com/questions/610291/how-to-install-cmake-3-2-on-ubuntu-14-04)

#### 3. Tegra X1

* gstreamer and opencv is already installed on TX1

```
sudo apt-get update
sudo apt-get install -y cmake libglib2.0-dev \
    libgoogle-glog-dev libboost-all-dev libjemalloc-dev libzmq3-dev \
    libeigen3-dev
# Optinal: install cnmem
git clone https://github.com/NVIDIA/cnmem
cd cnmem
mkdir build
cd build
cmake ..
make
sudo make install
```


### Compile

* `git clone https://github.com/ranxian/streamer`
* `mkdir build`
* `cd build`
* `cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_HOME=/path/to/caffe -DUSE_CAFFE=true -DBACKEND=cuda ..` (this is an example of building with Caffe on a machine supports CUDA). If you don't have a GPU from NVIDIA, you could specify `-DBACKEND=cpu`.
* `make`

To run unit tests:

* `cmake -DTEST_ON=true ..`
* `make tests`
* `ctest -j4`

### Run multi-camera end-to-end demo

This is an example of running imagenet netowrk in Caffe with multiple camera streams.

First you need to configure your cameras and models. In your build directory, there should be a `config` directory.

#### Configure cameras
1. `cp config/cameras.toml.example config/cameras.toml`.
2. Edit `config/cameras.toml`, you need to fill in the `name` of the camera, and the `video_uri` of the camera. `video_uri` could be
    * `rtsp://xxx`: An rtsp endpoint.
    * `facetime`: The iSight camera available on Mac.
    * `gst://xxx`: xxx could be any raw GStreamer video pipeline, as long as the pipleine emits valid video frames. For example `gst://videotestsrc`. streamer uses GStreamer heavily for video ingestion, please refer to https://gstreamer.freedesktop.org/ for more information.
3 `height`: Height of the camera.
4 `width`: Width of the camera.
    
    
#### Configure models
1. `cp config/models.toml.example config/models.toml`
2. Edit `config/models.toml`, fill in various fields for each of your model.
    * `name`: name of the model
    * `type`: either `"caffe"`, `"mxnet"` or `"gie"`
    * `desc_path`: The path to the model description file. For caffe and GIE it is `.prototxt` file, for mxnet it is `.json` file
    * `params_path`: The path to the model weights. For caffe and GIE it is `.caffemodel` file, for mxnet it is `.params` file
    * `input_width`: The suggested width of the input image.
    * `input_height`: The suggested height of the input image.
    * `label_file`: this is optional. For classification network or other network that requires a label file to produce a string label, you may add the path to that file here.

#### Configure encoders and decoders
The default configuration uses software encoder and decoder, to specify other decoders and encoders available on the system, edit `config/config.toml.example`.
    
    
#### Run the classification demo    

```
export LD_LIBRARY=$LD_LIBRARY:/path/to/caffe/lib
apps/multicam -h
Multi-camera end to end video ingestion demo:
  -h [ --help ]                  print the help message
  -m [ --model ] MODEL           The name of the model to run
  -c [ --camera ] CAMERAS        The name of the camera to use, if there are
                                 multiple cameras to be used, separate with ,
  -d [ --display ]               Enable display or not
  -C [ --config_dir ] CONFIG_DIR The directory to find streamer's configuations
```

Use `apps/multicam -h` to show helps. For example to run AlexNet on Mac's iSight built-in camera:

```
export LD_LIBRARY=$LD_LIBRARY:/path/to/caffe/lib
apps/multicam -m AlexNet -c Facetime -d
```

## Run with different frameworks

Currently we support BVLC caffe, Caffe's OpenCL branch, NVIDIA caffe (including fp16 caffe), GIE (GPU Inference Engine) and MXNet. Different frameworks are supported through several cmake flags.
**Cmake will cache the variables**, when you switch to another framework, better clean existed build.

* Caffe
	* Installation: http://caffe.berkeleyvision.org/installation.html
	* Build with: `cmake -DCAFFE_HOME=/path/to/caffe -DUSE_CAFFE=true ..`
* Caffe OpenCL
    * Installation`git clone https://github.com/BVLC/caffe/tree/opencl`
    * I can successfully built it with cmake, the steps are
    * `mkdir build`
    * `cmake ..`, check that `-DCPU_ONLY=off` if you want to use OpenCL
    * `make -j && make install`, built targets should be in `build/install`
    * Then build `streamer` with `cmake -DCAFFE_HOME=/path/to/opencl-caffe -DBACKEND=opencl`
    * Notice that OpenCL caffe has a different interface than the original caffe.
* NVDIA fp16 Caffe
	* Installation: install from https://github.com/NVIDIA/caffe/tree/experimental/fp16
	* Build with: `cmake -DCAFFE_HOME=/path/to/fp16caffe -DUSE_CAFFE=true -DUSE_FP16=true -DBACKEND=cuda ..`
* GIE
	* Installation: Apply usage through NVIDIA developer program: https://developer.nvidia.com/tensorrt
	* Build with: `cmake -DGIE_HOME=/path/to/gie -DUSE_GIE=true -DBACKEND=cuda ..`
	* GIE must be built with `-DBACKEND=cuda`. Also note that GIE can't be built together with Caffe.
* MXNet
	* Installation: http://mxnet.readthedocs.io/en/latest/how_to/build.html
	* Build with: `cmake -DMXNET_HOME=/path/to/mxnet -DUSE_MXNET=true ..`

## Compile configurations

Apart from various configuations for different frameworks:

* Control the backend
    1. Use CPU. `-DBACKEND=cpu`
    2. Use CUDA device. `-DBACKEND=cuda`
    3. Use OpenCL device. `-DBACKEND=opencl`
    
* Build with different framework: See above sections

* Build with PtGray SDK for their GigE cameras: `-DUSE_PTGRAY=on`

## TODO

* [x] Refactor cmake files
* [x] Port to CPU
* [x] Support Intel integrated gpu
* [x] Tegra as a streaming hub
* [ ] Video and metadata storage
* [ ] Support TensorFlow


