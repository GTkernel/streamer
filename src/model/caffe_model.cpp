//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "caffe_model.h"
#include <streamer.h>

template <typename DType>
CaffeModel<DType>::CaffeModel(const ModelDesc &model_desc, Shape input_shape,
                              int batch_size)
    : Model(model_desc, input_shape, batch_size) {}

template <typename DType>
void CaffeModel<DType>::Load() {
  // Set Caffe backend
  int desired_device_number = Context::GetContext().GetInt(DEVICE_NUMBER);

  if (desired_device_number == DEVICE_NUMBER_CPU_ONLY) {
    LOG(INFO) << "Use device: " << desired_device_number << "(CPU)";
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  } else {
#ifdef USE_CUDA
    std::vector<int> gpus;
    GetCUDAGpus(gpus);

    if (desired_device_number < gpus.size()) {
      // Device exists
      LOG(INFO) << "Use GPU with device ID " << desired_device_number;
      caffe::Caffe::SetDevice(desired_device_number);
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
    } else {
      LOG(FATAL) << "No GPU device: " << desired_device_number;
    }
#elif USE_OPENCL
    std::vector<int> gpus;
    int count = caffe::Caffe::EnumerateDevices();

    if (desired_device_number < count) {
      // Device exists
      LOG(INFO) << "Use GPU with device ID " << desired_device_number;
      caffe::Caffe::SetDevice(desired_device_number);
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
    } else {
      LOG(FATAL) << "No GPU device: " << desired_device_number;
    }
#else
    LOG(FATAL) << "Compiled in CPU_ONLY mode but have a device number "
                  "configured rather than -1";
#endif
  }

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
  input_layer->Reshape(batch_size_, input_shape_.channel, input_shape_.height,
                       input_shape_.width);
  // Forward dimension change to all layers.
  net_->Reshape();
  // Prepare input buffer
  input_layer = net_->input_blobs()[0];
  DType *input_data = input_layer->mutable_cpu_data();

  input_buffer_ = DataBuffer(
      input_data, batch_size_ * input_shape_.GetSize() * sizeof(DType));
}

template <typename DType>
void CaffeModel<DType>::Forward() {
  net_->Forward();
}

template <typename DType>
void CaffeModel<DType>::Evaluate() {
  output_shapes_.clear();
  output_buffers_.clear();

  CHECK(net_->input_blobs()[0]->mutable_cpu_data() ==
        input_buffer_.GetBuffer());

  net_->Forward();
  // Copy the output of the network
  const auto &output_blobs = net_->output_blobs();
  // TODO: consider doing it lazily, e.g. when we actually retrieve the output
  // data
  for (const auto &output_blob : output_blobs) {
    const DType *output_data = output_blob->mutable_cpu_data();
    Shape shape(output_blob->channels(), output_blob->width(),
                output_blob->height());
    output_shapes_.push_back(shape);
    DataBuffer output_buffer(batch_size_ * shape.GetSize() * sizeof(DType));
    output_buffer.Clone(output_data,
                        batch_size_ * shape.GetSize() * sizeof(DType));
    output_buffers_.push_back(output_buffer);
  }
}

template class CaffeModel<float>;
