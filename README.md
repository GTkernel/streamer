## tx1dnn

[![GitHub license](https://img.shields.io/badge/license-apache-green.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://travis-ci.com/ranxian/tx1dnn.svg?token=QaYrj2g7p1xx7VjDqDzv&branch=master)](https://travis-ci.com/ranxian/tx1dnn)

This is a playground for running DNN on NVIDIA's TX1 platform. The simplest thing it can do right now is to consume live video frames from a camera and run your favorite neural network on the frames.

tx1dnn currently supports Caffe, Caffe FP16 (NVIDIA's fork), MXNet, TensorFlow, TensorRT (former GIE). I'm trying hard to compile TensorFlow on TX1.

## Get Started

### Dependencies

You may want to install various deep learning frameworks (Caffe, TensorFlow, etc.). You can refer to their documentation on how to install them on your platform. Here I will only include minimal requirements to successfully build the project.

#### 1. Mac


With homebrew:

```
brew install cmake glog glib gstreamer gst-plugins-base \
	gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-ffmpeg \
	boost opencv
```

#### 2. Linux x86 (Ubuntu)

```
sudo apt-get update
sudo apt-get install -y gstreamer1.0 cmake libglib2.0-dev \
	libgoogle-glog-dev libboost-all-dev libopencv-dev
```

#### 3. Tegra X1

* gstreamer and opencv is already installed on TX1

```
sudo apt-get update
sudo apt-get install -y cmake libglib2.0-dev \
	libgoogle-glog-dev libboost-all-dev
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

* `git clone https://github.com/ranxian/tx1dnn`
* `mkdir build`
* `cd build`
* `cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_HOME=/path/to/caffe -USE_CAFFE=true ..` (this is an example of building with Caffe)
* `make`

To run unit tests:
* `cmake -DTEST_ON=true`
* `make test`

### Run demo

This is an example of running classification network in Caffe.

First you need to config your cameras and models. In your build directory, there should be a `config` directory.

#### Configure cameras
1. `cp config/cameras.toml.example config/cameras.toml`
2. Edit `config/cameras.toml`, you need to fill the `name` of the camera, and the `video_uri` of the camera. `video_uri` could be
    * `rtsp://xxx`: An rtsp endpoint.
    * `facetime`: The iSight camera available on Mac.
    * `gst://xxx`: xxx could be any raw GStreamer video pipeline, as long as the pipleine emits valid video frames. For example `videotestsrc`
    
    
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
    
    
#### Run the classification demo    

```
export LD_LIBRARY=$LD_LIBRARY:/path/to/caffe/lib
apps/classify CAMERA MODEL DISPLAY?
```

* `CAMERA` is the name of the camera.
* `MODEL` is the name of the model.
* `DISPLAY?` is either true: enable preview, or false.

Use `apps/classify -h` to show helps.

## Run with different frameworks

Currently we support BVLC caffe, NVIDIA caffe (including fp16 caffe), GIE (GPU Inference Engine) and MXNet. Different frameworks are supported through several cmake flags.

* Caffe
	* Installation: http://caffe.berkeleyvision.org/installation.html
	* Build with: `cmake -DCAFFE_HOME=/path/to/caffe -DUSE_CAFFE=true ..`
* NVDIA fp16 Caffe
	* Installation: install from https://github.com/NVIDIA/caffe/tree/experimental/fp16
	* Build with: `cmake -DCAFFE_HOME=/path/to/fp16caffe -DUSE_CAFFE=true -DUSE_FP16=true ..`
* GIE
	* Installation: Apply through NVIDIA developer program: https://developer.nvidia.com/tensorrt
	* Build with: `cmake -DGIE_HOME=/path/to/gie -DUSE_GIE=true ..`
	* Note that GIE can't be built together with Caffe
* MXNet
	* Installation: http://mxnet.readthedocs.io/en/latest/how_to/build.html
	* Build with: `cmake -DMXNET_HOME=/path/to/mxnet -DUSE_MXNET=true ..`

## Compile configurations

Apart from various configuations for different frameworks:

* `-DCPU_ONLY`: Compile without NVIDIA GPU available.

## TODO

* [x] Refactor cmake files
* [x] Port to CPU
* [ ] Video and metadata storage
* [ ] Tegra as a streaming hub
* [ ] Support TensorFlow
* [ ] Support Intel integrated gpu
