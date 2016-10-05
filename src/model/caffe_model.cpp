//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "caffe_model.h"
#include "utils/utils.h"

template <typename DType>
CaffeModel<DType>::CaffeModel(const ModelDesc &model_desc, Shape input_shape)
    : Model(model_desc, input_shape) {}

template <typename DType>
void CaffeModel<DType>::Load() {
	// Set Caffe backend
#ifdef USE_CAFFE
#ifdef USE_CUDA
  std::vector<int> gpus;
  GetCUDAGpus(gpus);

  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    caffe::Caffe::SetDevice(gpus[0]);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
#endif
#ifdef USE_OPENCL
  std::vector<int> gpus;
  int count = caffe::Caffe::EnumerateDevices();
  for (int i = 0; i < count; i++) {
    gpus.push_back(i);
  }

  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << 0;
    caffe::Caffe::SetDevice(1);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
#endif
#ifdef CPU_ONLY
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
#endif
#endif

// Load the network.
#ifdef USE_OPENCL
  net_.reset(new caffe::Net<DType>(model_desc_.GetModelDescPath(), caffe::TEST,
                                   caffe::Caffe::GetDefaultDevice()));
#else
  net_.reset(
      new caffe::Net<DType>(model_desc_.GetModelDescPath(), caffe::TEST));
#endif
  net_->CopyTrainedLayersFrom(model_desc_.GetModelParamsPath());

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  CHECK(input_shape_.channel == 3 || input_shape_.channel == 1)
      << "Input layer should have 1 or 3 channels.";

  caffe::Blob<DType> *input_layer = net_->input_blobs()[0];
  // Adjust input dimensions
  input_layer->Reshape(1, input_shape_.channel, input_shape_.height,
                       input_shape_.height);
  // Forward dimension change to all layers.
  net_->Reshape();
  // Prepare input buffer
  input_layer = net_->input_blobs()[0];
  DType *input_data = input_layer->mutable_cpu_data();

  input_buffer_ =
      DataBuffer(input_data, input_shape_.GetSize() * sizeof(DType));
}

template <typename DType>
void CaffeModel<DType>::Evaluate() {
  output_shapes_.clear();
  output_buffers_.clear();

  CHECK(net_->input_blobs()[0]->mutable_cpu_data() ==
        input_buffer_.GetBuffer());

  net_->Forward();
  // Copy the output of the network
  auto output_blobs = net_->output_blobs();
  // TODO: consider doing it lazily, e.g. when we actually retrieve the output
  // data
  for (auto output_blob : output_blobs) {
    DType *output_data = output_blob->mutable_cpu_data();
    Shape shape(output_blob->channels(), output_blob->width(),
                output_blob->height());
    output_shapes_.push_back(shape);
    DataBuffer output_buffer(shape.GetSize() * sizeof(DType));
    output_buffer.Clone(output_data, shape.GetSize() * sizeof(DType));
    output_buffers_.push_back(
        DataBuffer(output_data, shape.GetSize() * sizeof(DType)));
  }
}

template class CaffeModel<float>;
