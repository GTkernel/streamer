## tx1dnn

This is a playground for running DNN on NVIDIA's TX1 platform. The project is currently only going to be built and run on Tegra X1, the code will be ported to intel CPU when I have time and interests. 

## Get Started

To compile:

* `mkdir build`
* `cd build`
* `cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_HOME=/path/to/caffe -DGIE_HOME=/path/to/GIE -DMXNET_HOME=/path/to/mxnet -DUSE_GIE=False -DENABLE_FP16=True ..`
* `make`

Run:
```
apps/tx1_classify \
    ~/Code/caffe/models/bvlc_alexnet/deploy.prototxt \
    ~/Code/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel \
    ~/Code/caffe/data/ilsvrc12/imagenet_mean.binaryproto \
    ~/Code/caffe/data/ilsvrc12/synset_words.txt \
    'rtsp://user:passwd@camera1/cam/realmonitor?channel=1&subtype=1'` \
    false \
    caffe
```

## Run with different frameworks

Currently we support BVLC caffe, NVIDIA caffe (including fp16 caffe), GIE (GPU Inference Engine) and MXNet. Because GIE and Caffe are not compatible with each other, and BVLC caffe and NVIDIA have different API, some compile time configurations needs to be set in order to use different frameworks.

* Run with BVLC Caffe
  `cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_HOME=/path/to/caffe/distribute -DGIE_HOME=/path/to/GIE -DMXNET_HOME=/path/to/mxnet -DUSE_GIE=False -DENABLE_FP16=False ..`
* Run with NVIDIA fp16 Caffe
  `cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_HOME=/path/to/caffe/distribute -DGIE_HOME=/path/to/GIE -DMXNET_HOME=/path/to/mxnet -DUSE_GIE=False -DENABLE_FP16=True ..`
* Run with GIE
  `cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_HOME=/path/to/caffe/distribute -DGIE_HOME=/path/to/GIE -DMXNET_HOME=/path/to/mxnet -DUSE_GIE=True -DENABLE_FP16=False ..`
* Run with MXNet: MXNet can be run with either of the above cmake command

## TODO

[ ] Refactor cmake files
[ ] Port to CPU