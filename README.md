## Streamer Overview

[![GitHub license](https://img.shields.io/badge/license-apache-green.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

Streamer is a platform for real-time video ingestion and analytics, primarily of
live camera streams, with a special focus on running state-of-the-art machine
learning-based vision algorithms. The goal of streamer is twofold:

* An edge-to-cloud compute system that deploys real-time machine learning-based
computer vision applications across many geo-distributed cameras.
* An expressive pipeline-based programming model that makes it easy to define
complex applications the operate on multitudes of camera streams.

An example end-to-end ingestion and analytics pipeline:
* Take frames from cameras over RTSP.
* Decode the stream with a hardware H.264 decoder.
* Transform, scale, and normalize the frames into BGR images that can be fed
into a deep neural network.
* Run an image classification deep neural network (e.g. GoogleNet, MobileNet).
* Collect the classification results as well as the original video data and
store them locally on disk.

Streamer currently supports these DNN frameworks:
* [BVLC Caffe](https://github.com/BVLC/caffe)
* [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl)
* [Intel Caffe](https://github.com/intel/caffe)
* [TensorFlow](https://github.com/tensorflow/tensorflow)

## Quick Start

See [these detailed instructions](INSTALL.md) for more information about
installing Streamer and its dependencies.

### 1. Compile Streamer
```sh
git clone https://github.com/viscloud/streamer
cd streamer
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBACKEND=cpu -DTEST_ON=yes ..
make
make tests
make test
```

### 2. View frames from a camera

The `simple` example app displays frames from a "camera" on the screen. A few
examples are included below. See the `config/cameras.toml` file for more
details.

**Note 1:** The `--display` option requires X11, but the `simple` app works without
it and will display some basic frame statistics on the terminal.

**Note 2:** By default, the "file" camera reads from a file `video.mp4` in the
current directory (see `cameras.toml`).

```sh
make apps
./apps/simple --display --camera GST_TEST
./apps/simple --display --camera webcam-linux
./apps/simple --display --camera webcam-mac
./apps/simple --display --camera file
```

### 3. Run basic image classification

This is an example of running image classification on a single camera stream.

#### Add Caffe support to Streamer

First, we need to recompile Streamer with Caffe support. If Caffe is installed
globally, then you can omit `-DCAFFE_HOME`. The `cmake` command below assumes
that you have compiled Caffe using its offical `make` build system, not its
unofficial `cmake` system.
```sh
cmake -DUSE_CAFFE=yes -DCAFFE_HOME=/path/to/caffe/distribute ..
make apps
```

#### Configure the models

Next, we need to configure our models. In your build directory, there should be
a `config` directory. The example `models.toml` file has configurations for
using Caffe to perform image classification using GoogleNet and MobileNet. Note
that some of the parameters (e.g., the ones starting with `input_`) are model
specific.

The `models.toml` file also has the URLs for the various model files. There are
three files for each of the two models, but one of the files is shared--thus a
total of five files for both GoogleNet and MobileNet. Download the model files:
```sh
mkdir ../models
# Download the model files into ../models
```

#### Run the classification demo

```sh
./apps/classifier --display --camera webcam-linux --model googlenet
./apps/classifier --display --camera webcam-linux --model mobilenet
```

The `classifier` app also works without the `--display` option.
