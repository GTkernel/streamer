//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "model/caffe_model.h"

#include "common/context.h"
#include "model/model_manager.h"

template <typename DType>
CaffeModel<DType>::CaffeModel(const ModelDesc& model_desc, Shape input_shape)
    : Model(model_desc, input_shape) {}

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
#endif  // USE_OPENCL
  }

// Load the network.
#ifdef USE_OPENCL
  net_.reset(new caffe::Net<DType>(model_desc_.GetModelDescPath(), caffe::TEST,
                                   caffe::Caffe::GetDefaultDevice()));
#else
  net_.reset(
      new caffe::Net<DType>(model_desc_.GetModelDescPath(), caffe::TEST));
#endif  // USE_OPENCL
  net_->CopyTrainedLayersFrom(model_desc_.GetModelParamsPath());

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  CHECK(input_shape_.channel == 3 || input_shape_.channel == 1)
      << "Input layer should have 1 or 3 channels.";

  caffe::Blob<DType>* input_layer = net_->input_blobs().at(0);
  // Adjust input dimensions
  // batch size is enforced to be 1
  input_layer->Reshape(1, input_shape_.channel, input_shape_.height,
                       input_shape_.width);
  // Forward dimension change to all layers.
  net_->Reshape();
}

template <typename DType>
std::unordered_map<std::string, cv::Mat> CaffeModel<DType>::Evaluate(
    const std::unordered_map<std::string, cv::Mat>& input_map,
    const std::vector<std::string>& output_layer_names) {
  CHECK_EQ(input_map.size(), 1)
      << "For Caffe models, exactly one input must be provided.";
  // There is only one value in the input map, and the second entry in the pair
  // is the input data.
  auto input = input_map.begin()->second;

  // Subtract the mean image
  int format;
  if (input_shape_.channel == 3) {
    format = CV_32FC3;
  } else {
    format = CV_32FC1;
  }
  cv::Scalar mean_colors = ModelManager::GetInstance().GetMeanColors();
  cv::Mat mean_image = cv::Mat(
      cv::Size(input_shape_.width, input_shape_.height), format, mean_colors);
  cv::Mat input_normalized;
  cv::subtract(input, mean_image, input_normalized);

  // Format the input data in the way that Caffe expects
  auto input_layer = net_->input_blobs().at(0);
  DType* data = input_layer->mutable_cpu_data();
  // This loop creates a cv::Mat for each channel that is configured to point to
  // a particular location in "data", but the data itself is not populated until
  // the call to cv::split(). output_channels points to the correct offsets in
  // the Caffe input blob
  std::vector<cv::Mat> output_channels;
  for (decltype(input_shape_.channel) j = 0; j < input_shape_.channel; ++j) {
    cv::Mat channel(input_shape_.height, input_shape_.width, CV_32F, data);
    output_channels.push_back(channel);
    data += input_shape_.width * input_shape_.height;
  }
  cv::split(input_normalized, output_channels);

  // Evaluate model on input
  net_->Forward();

  // Grab all the output layers
  std::unordered_map<std::string, cv::Mat> output_layers;
  for (const auto& layer : output_layer_names) {
    output_layers[layer] = GetLayerOutput(layer);
  }
  return output_layers;
}

template <typename DType>
void CaffeModel<DType>::Forward() {
  net_->Forward();
}

template <typename DType>
cv::Mat CaffeModel<DType>::GetLayerOutput(const std::string& layer_name) const {
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
    return BlobToMat2d(myblob);
  } else if (myblob->num_axes() == 4) {
    return BlobToMat4d(myblob);
  } else {
    LOG(FATAL)
        << "Error, only 2D and 4D feature vectors are supported at this time";
  }
}

template <typename DType>
cv::Mat CaffeModel<DType>::BlobToMat2d(caffe::Blob<DType>* src) {
  int batch_size = src->shape(0);
  CHECK(batch_size == 1) << "Batch size must be 1, but it is " << batch_size;

  std::vector<int> mat_size;
  for (decltype(src->num_axes()) i = 0; i < src->num_axes(); ++i) {
    mat_size.push_back(src->shape(i));
  }
  cv::Mat ret_mat(mat_size, CV_32F);
  // idxs contains the current 2-dimensional coordinate within the matrix.
  std::vector<int> idxs(2);
  // Batching has been disabled, so the batch index is always 0.
  idxs[0] = 0;

  auto x_size = mat_size.at(1);
  for (decltype(x_size) i = 0; i < x_size; ++i) {
    idxs[1] = i;
    ret_mat.at<DType>(0, i) = src->data_at(idxs);
  }
  return ret_mat;
}

template <typename DType>
cv::Mat CaffeModel<DType>::BlobToMat4d(caffe::Blob<DType>* src) {
  int batch_size = src->shape(0);
  CHECK(batch_size == 1) << "Batch size must be 1, but it is " << batch_size;

  // mat_size holds the axes dimensions of the blob
  // mat_size is used to construct the cv::Mat
  std::vector<int> mat_size;
  for (decltype(src->num_axes()) i = 0; i < src->num_axes(); ++i) {
    mat_size.push_back(src->shape(i));
  }
  cv::Mat ret_mat(mat_size, CV_32F);

  // idxs holds the current 4-dimensional coordinate within the matrix.
  std::vector<int> idxs(4);
  // Batching has been disabled, so the batch index is always 0.
  idxs[0] = 0;

  auto x_size = mat_size.at(1);
  for (decltype(x_size) i = 0; i < x_size; ++i) {
    idxs[1] = i;

    auto y_size = mat_size.at(2);
    for (decltype(y_size) j = 0; j < y_size; ++j) {
      idxs[2] = j;

      auto z_size = mat_size.at(3);
      for (decltype(z_size) k = 0; k < z_size; ++k) {
        idxs[3] = k;
        ret_mat.at<DType>(idxs.data()) = src->data_at(idxs);
      }
    }
  }
  return ret_mat;
}

template class CaffeModel<float>;
