//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "caffe_fp16_model.h"
#include "common/context.h"
#include "utils/utils.h"

CaffeFp16Model::CaffeFp16Model(const ModelDesc& model_desc, Shape input_shape,
                               int batch_size)
    : Model(model_desc, input_shape, batch_size) {}

void CaffeFp16Model::Load() {
  // Set Caffe backend
  int desired_device_number = Context::GetContext().GetInt(DEVICE_NUMBER);

  if (desired_device_number == DEVICE_NUMBER_CPU_ONLY) {
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
#else
    LOG(FATAL) << "Compiled in CPU_ONLY mode but have a device number "
                  "configured rather than -1";
#endif  // USE_CUDA
  }
  // Load the network.
  net_.reset(new caffe::Net<DType, MType>(model_desc_.GetModelDescPath(),
                                          caffe::TEST));
  net_->CopyTrainedLayersFrom(model_desc_.GetModelParamsPath());

  CHECK(input_shape_.channel == 3 || input_shape_.channel == 1)
      << "Input layer should have 1 or 3 channels.";

  caffe::Blob<DType, MType>* input_layer = net_->input_blobs()[0];

  // Adjust input dimensions
  input_layer->Reshape(batch_size_, input_shape_.channel, input_shape_.height,
                       input_shape_.width);

  // Forward dimension change to all layers.
  net_->Reshape();

  // Prepare input buffer
  input_buffer_ =
      DataBuffer(input_shape_.GetSize() * sizeof(float) * batch_size_);

  DType* input_data = input_layer->mutable_cpu_data();
  network_input_buffer_ = DataBuffer(
      input_data, input_shape_.GetSize() * sizeof(DType) * batch_size_);
}

void CaffeFp16Model::Forward() { net_->ForwardPrefilled(); }

void CaffeFp16Model::Evaluate() {
  // Copy the input to half precision  input
  Timer timer;
  timer.Start();
  if (sizeof(DType) == 2) {
    float* fp32data = (float*)input_buffer_.GetBuffer();
    DType* fp16data = (DType*)(network_input_buffer_.GetBuffer());

    size_t image_size = input_shape_.GetSize() * batch_size_;
    for (decltype(image_size) i = 0; i < image_size; ++i) {
      fp16data[i] = caffe::Get<DType>(fp32data[i]);
    }
  } else {
    LOG(WARNING) << "Clone partial buffer data";
    DataBuffer temp_buffer(input_buffer_.GetBuffer(),
                           network_input_buffer_.GetSize());
    network_input_buffer_.Clone(temp_buffer);
  }

  LOG(INFO) << "transform input took " << timer.ElapsedMSec() << " ms";

  // Evaluate
  output_shapes_.clear();
  output_buffers_.clear();

  net_->ForwardPrefilled();

  // Copy the output of the network
  auto output_blobs = net_->output_blobs();
  for (const auto& output_blob : output_blobs) {
    Shape shape(output_blob->channels(), output_blob->width(),
                output_blob->height());
    DataBuffer output_buffer;
    timer.Start();
    if (sizeof(DType) == 2) {
      output_buffer = DataBuffer(shape.GetSize() * sizeof(float) * batch_size_);
      float* fp32data = (float*)output_buffer.GetBuffer();
      DType* fp16data = output_blob->mutable_cpu_data();
      size_t len = shape.GetSize() * batch_size_;
      for (decltype(len) i = 0; i < len; ++i) {
        fp32data[i] = caffe::Get<float>(fp16data[i]);
      }
    } else {
      output_buffer = DataBuffer(output_blob->mutable_cpu_data(),
                                 shape.GetSize() * sizeof(float) * batch_size_);
    }
    LOG(INFO) << "transform output took " << timer.ElapsedMSec() << " ms";

    output_shapes_.push_back(shape);
    output_buffers_.push_back(output_buffer);
  }
}

const std::vector<std::string>& CaffeFp16Model::GetLayerNames() const {
  STREAMER_NOT_IMPLEMENTED;
}

cv::Mat CaffeFp16Model::GetLayerOutput(const std::string&) const {
  STREAMER_NOT_IMPLEMENTED;
}
