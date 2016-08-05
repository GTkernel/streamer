//
// Created by Ran Xian on 8/1/16.
//

#include "caffe_v1_classifier.h"
#include "utils.h"

template <typename DType>
CaffeV1Classifier<DType>::CaffeV1Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file): Classifier(label_file) {
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
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
  << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  // Load the binaryproto mean file.
  SetMean(mean_file);


  caffe::Blob<DType>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";

  // Adjust input dimensions
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
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
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
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
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

template <typename DType>
std::vector<float> CaffeV1Classifier<DType>::Predict(const cv::Mat& img) {
  Timer timerTotal;
  timerTotal.Start();
  Timer timer;
  timer.Start();

  Preprocess(img);
  LOG(INFO) << "Preprocessing done in " << timer.ElapsedMSec() << "ms";

  timer.Start();
  net_->Forward();
  LOG(INFO) << "Forward done in " << timer.ElapsedMSec() << "ms";

  /* Copy the output layer to a std::vector */
  
  std::vector<float> scores;
  caffe::Blob<DType>* output_layer = net_->output_blobs()[0];
  const DType* begin = output_layer->cpu_data();
  timer.Start();
  const DType* end = begin + output_layer->channels();
  for (const DType *ptr = begin; ptr != end; ptr++) {
    scores.push_back(*ptr);
  }
  LOG(INFO) << "Copied output layer in " << timer.ElapsedMSec() << "ms";
  LOG(INFO) << "Whole predict done in " << timerTotal.ElapsedMSec() << "ms";
  return scores;
}

/**
 * @brief This function will transform the input data to proper forms to be fed into the model. It will directly write
 * the input data to input layers.
 * @param img The input image.
 * @param input_channels
 */
template <typename DType>
void CaffeV1Classifier<DType>::Preprocess(const cv::Mat& img) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample_transformed = TransformImage(img, num_channels_, input_geometry_.width, input_geometry_.height);

  cv::Mat sample_normalized;
  cv::subtract(sample_transformed , mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  std::vector<cv::Mat> split_channels;
  split_channels.resize(num_channels_);
  cv::split(sample_normalized, split_channels);

  // Get the input channel
  caffe::Blob<DType>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  DType *input_data = input_layer->mutable_cpu_data();
  DType *input_data_ptr = input_data;

  // Convert fp32 to fp16, and fill in input channels one by one
  for (int i = 0; i < num_channels_; i++) {
    memcpy(input_data_ptr, split_channels[i].data, input_geometry_.area() * sizeof(DType));
    input_data_ptr += input_geometry_.area();
  }
}

template class CaffeV1Classifier<float>;
