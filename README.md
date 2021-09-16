# Streamer

This is legancy project of [SAF](https://github.com/viscloud/saf). 
There is no plan for maintainess and feature updates.
It original README is at [here](https://github.com/GTKernel/streamer/blob/master/OFFICIAL_README.md).
This README shows you the customized implementation and deployment.

Streamer is also used as the ML inference application in [Couper](https://github.com/GTkernel/couper).

## How to start

Build building and running streamer, please prefare your pre-trained TensorFlow model in Protocol Buffer format.
And put the model under directory `./models`.
### Build and Run

### Build the docker images

First, build the builder images based on GPU or not under `./docker/tf_env`. 
For, CPU-only host, using file `Dockerfile_cpu`; if you have Nvidia GPU, using file `Dockerfile`.

Next, build streamer image by the Dockerfile under root directory.
Be awared to change the image name of builder in the file.

### Run container in K8s
Please refer to this one in Couper [repo](https://github.com/GTkernel/couper/blob/master/saf_pipeline.yml).

## Support

Welcome to update and discuss the usecases througth issue tickets and pull requests.
Or, for any other question, please contact to [Carol Hsu](mailto:nosus_hsu@gatech.edu).
