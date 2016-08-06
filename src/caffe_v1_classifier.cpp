//
// Created by Ran Xian on 8/1/16.
//

#include "caffe_v1_classifier.h"
#include "utils.h"

template <typename DType>
CaffeV1Classifier<DType>::CaffeV1Classifier(const string& model_file,
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
  net_.reset(new caffe::Net<DType>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  caffe::Blob<DType>* input_layer = net_->input_blobs()[0];
  input_channels_ = input_layer->channels();
  CHECK(input_channels_ == 3 || input_channels_ == 1)
  << "Input layer should have 1 or 3 channels.";
  input_width_ = input_layer->width();
  input_height_ = input_layer->height();

  // Load the binaryproto mean file.
  SetMean(mean_file);


  caffe::Blob<DType>* output_layer = net_->output_blobs()[0];

  // Adjust input dimensions
  input_layer->Reshape(1, input_channels_, input_height_, input_width_);
  // Forward dimension change to all layers.
  net_->Reshape();
}

/* Load the mean file in binaryproto format. */
template <typename DType>
void CaffeV1Classifier<DType>::SetMean(const string& mean_file) {
  caffe::BlobProto blob_proto;
  caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  caffe::Blob<float> mean_blob;
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

template <typename DType>
std::vector<float> CaffeV1Classifier<DType>::Predict() {
  Timer timer; timer.Start();

  net_->Forward();
  // Copy the output layer to a std::vector
  caffe::Blob<DType>* output_layer = net_->output_blobs()[0];
  const DType* begin = output_layer->cpu_data();
  LOG(INFO) << "Forward done in " << timer.ElapsedMSec() << "ms";

  std::vector<float> scores;
  const DType* end = begin + output_layer->channels();
  for (const DType *ptr = begin; ptr != end; ptr++) {
    scores.push_back(*ptr);
  }
  return scores;
}

template <typename DType>
DataBuffer CaffeV1Classifier<DType>::GetInputBuffer() {
  caffe::Blob<DType>* input_layer = net_->input_blobs()[0];
  DType* input_data = input_layer->mutable_cpu_data();

  DataBuffer buffer(input_data, GetInputSize<DType>());

  return buffer;
}

template class CaffeV1Classifier<float>;
