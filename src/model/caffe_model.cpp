//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "model/caffe_model.h"

#include "common/context.h"
#include "model/model_manager.h"

template <typename DType>
CaffeModel<DType>::CaffeModel(const ModelDesc& model_desc, Shape input_shape,
                              size_t batch_size)
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

    if (desired_device_number < (int)gpus.size()) {
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
#endif  // USE_OPENCL
  }

// Load the network.
#ifdef USE_OPENCL
  net_ = std::make_unique<caffe::Net<DType>>(model_desc_.GetModelDescPath(), caffe::TEST,
                                   caffe::Caffe::GetDefaultDevice()));
#else
  net_ = std::make_unique<caffe::Net<DType>>(model_desc_.GetModelDescPath(),
                                             caffe::TEST);
#endif  // USE_OPENCL
  net_->CopyTrainedLayersFrom(model_desc_.GetModelParamsPath());

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  CHECK(input_shape_.channel == 3 || input_shape_.channel == 1)
      << "Input layer should have 1 or 3 channels.";

  caffe::Blob<DType>* input_layer = net_->input_blobs().at(0);
  input_layer->Reshape(batch_size_, input_shape_.channel, input_shape_.height,
                       input_shape_.width);
  // Forward dimension change to all layers.
  net_->Reshape();
}

template <typename DType>
cv::Mat CaffeModel<DType>::ConvertAndNormalize(cv::Mat img) {
  int format;
  if (input_shape_.channel == 3) {
    format = CV_32FC3;
  } else {
    format = CV_32FC1;
  }

  // Convert from CV_8UCX to CV_32FCX
  cv::Mat input;
  img.convertTo(input, format);
  cv::Scalar mean_colors = ModelManager::GetInstance().GetMeanColors();
  cv::Mat mean_image = cv::Mat(
      cv::Size(input_shape_.width, input_shape_.height), format, mean_colors);
  // Subtract the mean image
  cv::Mat input_normalized(cv::Size(input_shape_.width, input_shape_.height),
                           format);
  cv::subtract(input, mean_image, input_normalized);
  input_normalized *= model_desc_.GetInputScale();
  return input_normalized;
}

template <typename DType>
std::unordered_map<std::string, std::vector<cv::Mat>>
CaffeModel<DType>::Evaluate(
    const std::unordered_map<std::string, std::vector<cv::Mat>>& input_map,
    const std::vector<std::string>& output_layer_names) {
  CHECK_EQ(input_map.size(), 1)
      << "For Caffe models, exactly one input must be provided.";
  // There is only one value in the input map, and the second entry in the pair
  // is the input data.
  CHECK_EQ(input_map.begin()->second.size(), batch_size_)
      << "Wrong batch size, "
      << "expected: " << batch_size_ << "found: " << input_map.at(0).size();

  caffe::Blob<DType>* input_layer = net_->input_blobs().at(0);
  DType* data = input_layer->mutable_cpu_data();
  for (const auto& input : input_map.begin()->second) {
    cv::Mat input_normalized = ConvertAndNormalize(input);

    // Format the input data in the way that Caffe expects
    // This loop creates a cv::Mat for each channel that is configured to point
    // to a particular location in "data", but the data itself is not populated
    // until the call to cv::split(). output_channels points to the correct
    // offsets in the Caffe input blob
    std::vector<cv::Mat> output_channels;

    for (decltype(input_shape_.channel) j = 0; j < input_shape_.channel; ++j) {
      cv::Mat channel(input_shape_.height, input_shape_.width, CV_32F, data);
      output_channels.push_back(channel);
      data += input_shape_.width * input_shape_.height;
    }
    cv::split(input_normalized, output_channels);
  }

  // Evaluate model on input
  net_->Forward();

  // Grab all the output layers
  std::unordered_map<std::string, std::vector<cv::Mat>> output_layers;
  for (const auto& layer : output_layer_names) {
    for (decltype(batch_size_) batch_idx = 0; batch_idx < batch_size_;
         ++batch_idx) {
      output_layers[layer].push_back(GetLayerOutput(layer, batch_idx));
    }
  }
  return output_layers;
}

template <typename DType>
cv::Mat CaffeModel<DType>::GetLayerOutput(const std::string& layer_name,
                                          int batch_idx) const {
  const std::vector<std::vector<caffe::Blob<DType>*>> layer_outputs =
      net_->top_vecs();
  // Find the correct layer to extract
  std::vector<std::string> layer_names = net_->layer_names();
  auto layer_idx =
      std::find(layer_names.begin(), layer_names.end(), layer_name);
  if (layer_idx == layer_names.end()) {
    LOG(FATAL) << "Layer \"" << layer_name << "\" does not exist";
  }
  int idx = layer_idx - layer_names.begin();
  caffe::Blob<DType>* myblob = layer_outputs.at(idx).at(0);
  // The last layer is often 2-dimensional (batch, 1D array of probabilities)
  // Intermediate layers are always 4-dimensional
  if (myblob->num_axes() == 2) {
    return BlobToMat2d(myblob, batch_idx);
  } else if (myblob->num_axes() == 4) {
    return BlobToMat4d(myblob, batch_idx);
  } else {
    LOG(FATAL)
        << "Error, only 2D and 4D feature vectors are supported at this time";
  }
  return cv::Mat();
}

template <typename DType>
cv::Mat CaffeModel<DType>::BlobToMat2d(caffe::Blob<DType>* src,
                                       int batch_idx) const {
  decltype(batch_size_) batch_size = src->shape(0);
  CHECK(batch_size == batch_size_) << "Incorrect batch size";

  // mat_size holds the axes dimensions of the blob
  std::vector<int> mat_size;
  decltype(src->shape(0)) total_size = 1;
  for (decltype(src->num_axes()) i = 0; i < src->num_axes(); ++i) {
    mat_size.push_back(src->shape(i));
    total_size *= src->shape(i);
  }
  float* data = src->mutable_cpu_data();
  cv::Mat ret_mat(mat_size, CV_32F);
  memcpy(ret_mat.data, data + total_size * batch_idx,
         total_size * sizeof(float));
  return ret_mat;
}

template <typename DType>
cv::Mat CaffeModel<DType>::BlobToMat4d(caffe::Blob<DType>* src,
                                       int batch_idx) const {
  size_t batch_size = (size_t)src->shape(0);
  CHECK(batch_size == batch_size_) << "Incorrect batch size";
  int num_channel = src->shape(1);
  int height = src->shape(2);
  int width = src->shape(3);
  int total_size = height * width * num_channel;
  DType* data = src->mutable_cpu_data();
  if (num_channel > CV_CN_MAX) {
    LOG(WARNING) << "Caffe output channels exceeds CV_CN_MAX (" << num_channel
                 << " > " << CV_CN_MAX << ")";
    CHECK(height == 1 && width == 1)
        << "NHWC format must be disabled for matrices with more than "
        << CV_CN_MAX << " channels and height/width != 1.";
    cv::Mat ret_mat({num_channel, height, width}, CV_32F);
    memcpy(ret_mat.data, data + total_size * batch_idx,
           total_size * sizeof(DType));
    return ret_mat;
  }

  // Convert from CHW to HWC
  cv::Mat ret_mat({height, width, num_channel}, CV_32F);
  int per_channel_floats = height * width;
  int per_channel_bytes = per_channel_floats * sizeof(DType);
  std::vector<cv::Mat> channels;
  DType* cur_batch_data = data + (num_channel * per_channel_floats) * batch_idx;
  for (int i = 0; i < num_channel; ++i) {
    cv::Mat cur_channel(height, width, CV_32F);
    memcpy(cur_channel.data, cur_batch_data + per_channel_floats * i,
           per_channel_bytes);
    channels.push_back(cur_channel);
  }
  cv::merge(channels, ret_mat);

// Element-wise comparison
#ifdef MODE_VERIFY
  LOG(INFO) << "Checking output matrix of size: " << height << "x" << width
            << "x" << num_channel;
  for (int c = 0; c < num_channel; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        if (src->shape(1) <= CV_CN_MAX) {
          DType lhs = ret_mat.ptr<DType>(h)[w * num_channel + c];
          DType rhs = src->data_at(batch_idx, c, h, w);
          CHECK(lhs == rhs)
              << "At index <h: " << h << " w: " << w << " c: " << c
              << "> found: " << lhs << " expected: " << rhs;
        }
      }
    }
  }
#endif  // MODE_VERIFY
  return ret_mat;
}

template class CaffeModel<float>;
