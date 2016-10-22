## streamer

[![GitHub license](https://img.shields.io/badge/license-apache-green.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://travis-ci.com/ranxian/streamer.svg?token=QaYrj2g7p1xx7VjDqDzv&branch=master)](https://travis-ci.com/ranxian/streamer)

This is a playground for running DNN on nowdays powerful heterogeneous embedded platform. The simplest thing it can do right now is to consume live video frames from a camera and run your favorite neural network on the frames.

streamer currently supports Caffe, Caffe FP16 (NVIDIA's fork), Caffe OpenCL, MXNet, TensorRT (former GIE from NVIDIA).

## Get Started

### Dependencies

You may want to install various deep learning frameworks (Caffe, MXNet, etc.). You can refer to their documentation on how to install them on your platform. Here I will only include minimal requirements to successfully build the project.

#### 1. Mac


With homebrew:

```
brew install cmake glog glib gstreamer gst-plugins-base \
	gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-ffmpeg \
	boost opencv jemalloc
```

#### 2. Linux x86 (Ubuntu)

```
sudo apt-get update
sudo apt-get install -y cmake libglib2.0-dev libgoogle-glog-dev \
    libboost-all-dev libopencv-dev gstreamer1.0 libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev libjemalloc-dev
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
	libgoogle-glog-dev libboost-all-dev libjemalloc-dev
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
* `cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_HOME=/path/to/caffe -DUSE_CAFFE=true -DBACKEND=cuda ..` (this is an example of building with Caffe)
* `make`

To run unit tests:
* `cmake -DTEST_ON=true`
* `make test`

### Run demo

This is an example of running classification network in Caffe with multiple camera streams.

First you need to configure your cameras and models. In your build directory, there should be a `config` directory.

#### Configure cameras
1. `cp config/cameras.toml.example config/cameras.toml`
2. Edit `config/cameras.toml`, you need to fill the `name` of the camera, and the `video_uri` of the camera. `video_uri` could be
    * `rtsp://xxx`: An rtsp endpoint.
    * `facetime`: The iSight camera available on Mac.
    * `gst://xxx`: xxx could be any raw GStreamer video pipeline, as long as the pipleine emits valid video frames. For example `gst://videotestsrc`. streamer uses GStreamer heavily for video ingestion, please refer to https://gstreamer.freedesktop.org/ for more information.
    
    
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
apps/multicam CAMERA[,CAMERA2[,...]] MODEL DISPLAY?
```

* `CAMERA...` are comma separated list of names of camera.
* `MODEL` is the name of the model.
* `DISPLAY?` is either true: enable preview, or false.

Use `apps/multicam` to show helps.

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
* NVDIA fp16 Caffe
	* Installation: install from https://github.com/NVIDIA/caffe/tree/experimental/fp16
	* Build with: `cmake -DCAFFE_HOME=/path/to/fp16caffe -DUSE_CAFFE=true -DUSE_FP16=true -DBACKEND=cuda ..`
* GIE
	* Installation: Apply through NVIDIA developer program: https://developer.nvidia.com/tensorrt
	* Build with: `cmake -DGIE_HOME=/path/to/gie -DUSE_GIE=true -DBACKEND=cuda ..`
	* GIE must be built with `-DBACKEND=cuda`. Also note that GIE can't be built together with Caffe
* MXNet
	* Installation: http://mxnet.readthedocs.io/en/latest/how_to/build.html
	* Build with: `cmake -DMXNET_HOME=/path/to/mxnet -DUSE_MXNET=true ..`

## Compile configurations

Apart from various configuations for different frameworks:

* Control the backend
    1. Use cpu. `-DBACKEND=cpu`
    2. Use CUDA device. `-DBACKEND=cuda`
    3. Use OpenCL device. `-DBACKEND=opencl`

## TODO

* [x] Refactor cmake files
* [x] Port to CPU
* [x] Support Intel integrated gpu
* [ ] Video and metadata storage
* [ ] Tegra as a streaming hub
* [ ] Support TensorFlow

