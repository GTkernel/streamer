//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#include "model/caffe_model.h"
#include "common/context.h"

template <typename DType>
CaffeModel<DType>::CaffeModel(const ModelDesc& model_desc, Shape input_shape,
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
  input_layer->Reshape(batch_size_, input_shape_.channel, input_shape_.height,
                       input_shape_.width);
  // Forward dimension change to all layers.
  net_->Reshape();
  // Prepare input buffer
  input_layer = net_->input_blobs().at(0);
  DType* input_data = input_layer->mutable_cpu_data();

  input_buffer_ = DataBuffer(
      input_data, batch_size_ * input_shape_.GetSize() * sizeof(DType));
}

template <typename DType>
void CaffeModel<DType>::Evaluate() {
  output_shapes_.clear();
  output_buffers_.clear();

  CHECK(net_->input_blobs().at(0)->mutable_cpu_data() ==
        input_buffer_.GetBuffer());

  net_->Forward();
  // Copy the output of the network
  const auto& output_blobs = net_->output_blobs();
  // TODO: consider doing it lazily, e.g. when we actually retrieve the output
  // data
  for (const auto& output_blob : output_blobs) {
    const DType* output_data = output_blob->mutable_cpu_data();
    Shape shape(output_blob->channels(), output_blob->width(),
                output_blob->height());
    output_shapes_.push_back(shape);
    DataBuffer output_buffer(batch_size_ * shape.GetSize() * sizeof(DType));
    output_buffer.Clone(output_data,
                        batch_size_ * shape.GetSize() * sizeof(DType));
    output_buffers_.push_back(output_buffer);
  }
}

template <typename DType>
void CaffeModel<DType>::Forward() {
  net_->Forward();
}

template <typename DType>
const std::vector<std::string>& CaffeModel<DType>::GetLayerNames() const {
  return net_->layer_names();
}

template <typename DType>
cv::Mat CaffeModel<DType>::GetLayerOutput(const std::string& layer_name) const {
  const std::vector<std::vector<caffe::Blob<DType>*>> layer_outputs =
      net_->top_vecs();
  // Find the correct layer to extract
  std::vector<std::string> layer_names = GetLayerNames();
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
