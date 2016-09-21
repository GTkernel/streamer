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
# Install OpenCV following http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation
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

### Run

This is an example of running classification network in Caffe.

```
export LD_LIBRARY=$LD_LIBRARY:/path/to/caffe/lib
apps/tx1_classify \
    ~/Code/caffe/models/bvlc_alexnet/deploy.prototxt \
    ~/Code/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel \
    ~/Code/caffe/data/ilsvrc12/imagenet_mean.binaryproto \
    ~/Code/caffe/data/ilsvrc12/synset_words.txt \
    'VIDEO_URI' \
    false \
    caffe
```

`VIDEO_URI` could be: 

* `facetime`: The iSight camera available on Mac
* `rtsp://xxx`: A rtsp endpoint
* Any other valid gstreamer video pipeline, e.g. `videotestsrc`

Use `apps/tx1_classify -h` to show options.

## Run with different frameworks

Currently we support BVLC caffe, NVIDIA caffe (including fp16 caffe), GIE (GPU Inference Engine) and MXNet. Different frameworks are supported through several cmake flags.

* Caffe
	* Installation: http://caffe.berkeleyvision.org/installation.html
	* Build with: `cmake -DCAFFE_HOME=/path/to/caffe -DUSE_CAFFE ..`
* NVDIA fp16 Caffe
	* Installation: install from https://github.com/NVIDIA/caffe/tree/experimental/fp16
	* Build with: `cmake -DCAFFE_HOME=/path/to/fp16caffe -DUSE_CAFFE -DUSE_FP16 ..`
* GIE
	* Installation: Apply through NVIDIA developer program: https://developer.nvidia.com/tensorrt
	* Build with: `cmake -DGIE_HOME=/path/to/gie -DUSE_GIE ..`
	* Note that GIE can't be built together with Caffe
* MXNet
	* Installation: http://mxnet.readthedocs.io/en/latest/how_to/build.html
	* Build with: `cmake -DMXNET_HOME=/path/to/mxnet -DUSE_MXNET ..`

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
