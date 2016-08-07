//
// Created by Ran Xian on 7/23/16.
//

#include "caffe_fp16_classifier.h"
#include "utils.h"
#include "float16.h"

CaffeFp16Classifier::CaffeFp16Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file): Classifier(model_file, trained_file, mean_file, label_file) {
#ifdef CPU_ONLY
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  std::vector<int> gpus;
  GetGpus(gpus);

  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    caffe::Caffe::SetDevice(gpus[0]);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  CHECK(gpus.size() != 0);
#endif

  // Load the network.
  net_.reset(new caffe::Net<DType, MType>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  caffe::Blob<DType, MType>* input_layer = net_->input_blobs()[0];
  input_channels_ = input_layer->channels();
  CHECK(input_channels_ == 3 || input_channels_ == 1) << "Input layer should have 1 or 3 channels.";
  input_width_ = input_layer->width();
  input_height_ = input_layer->height();

  caffe::Blob<DType, MType>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";

  // Adjust input dimensions
  input_layer->Reshape(1, input_channels_, input_height_, input_width_);

  // Forward dimension change to all layers.
  net_->Reshape();

  // Load the binaryproto mean file.
  SetMean(mean_file);
}

/* Load the mean file in binaryproto format. */
void CaffeFp16Classifier::SetMean(const string& mean_file) {
  caffe::BlobProto blob_proto;
  caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  caffe::Blob<float,float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), input_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < input_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_image_ = cv::Mat(GetInputGeometry(), mean.type(), channel_mean);
}

/**
 * @brief Transform to fp16.
 */
void CaffeFp16Classifier::Preprocess(const cv::Mat &img, DataBuffer &buffer) {
  CHECK(buffer.GetSize() == GetInputSize<DType>()) << "Buffer size does not match";
  DataBuffer temp_buffer(GetInputSize<float>());
  cv::Mat transformed = TransformImage(img, GetInputShape(), mean_image_, &temp_buffer);

  float *fp32data = (float *)temp_buffer.GetBuffer();
  DType *fp16data = (DType *)buffer.GetBuffer();

  size_t image_size = GetInputShape().GetVolume();
  for (size_t i = 0; i < image_size; i++) {
    fp16data[i] = caffe::Get<caffe::float16>(fp32data[i]);
  }
}

std::vector<float> CaffeFp16Classifier::Predict() {
  Timer timer;
  timer.Start();
  net_->ForwardPrefilled();
  caffe::Blob<DType, MType>* output_layer = net_->output_blobs()[0];
  const DType* begin = output_layer->cpu_data();
  LOG(INFO) << "FP16 forward done in " << timer.ElapsedMSec() << " ms";
  std::vector<float> scores;
  timer.Start();
  const DType* end = begin + output_layer->channels();
  for (const DType *ptr = begin; ptr != end; ptr++) {
    scores.push_back(caffe::Get<float>(*ptr));
  }
  LOG(INFO) << "FP16 copied output in " << timer.ElapsedMSec() << " ms";
  return scores;
}

DataBuffer CaffeFp16Classifier::GetInputBuffer() {
  caffe::Blob<DType, MType> *input_layer = net_->input_blobs()[0];
  DType* input_data = input_layer->mutable_cpu_data();

  DataBuffer buffer(input_data, GetInputSize<DType>());

  return buffer;
}
