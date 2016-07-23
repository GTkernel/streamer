## tx1dnn

This is a playground for running DNN on NVDIA's TX1 platform.

## Get Started
To compile:

* `mkdir build`
* `cd build`
* `cmake -DCMAKE_BUILD_TYPE=Release -DCAFFE_HOME=/home/ubuntu/Code/caffe/distribute ..`
* `make`

Run:
* `apps/tx1_run_alexnet ~/Code/caffe/models/bvlc_alexnet/deploy.prototxt ~/Code/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel ~/Code/caffe/data/ilsvrc12/imagenet_mean.binaryproto ~/Code/caffe/data/ilsvrc12/synset_words.txt 'rtsp://***REMOVED***@camera1/cam/realmonitor?channel=1&subtype=1'`
