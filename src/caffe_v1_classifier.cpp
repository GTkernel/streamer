//
// Created by Ran Xian on 8/1/16.
//

#include "caffe_v1_classifier.h"
#include "utils.h"

template <typename DType>
CaffeV1Classifier<DType>::CaffeV1Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
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

  // Load labels.
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  caffe::Blob<DType>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";

  // Adjust input dimensions
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  // Forward dimension change to all layers.
  net_->Reshape();
}

/* Return the top N predictions. */
template <typename DType>
std::vector<Prediction> CaffeV1Classifier<DType>::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
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
  std::vector<DType *> input_channels;

  timer.Start();
  WrapInputLayer(input_channels);
  LOG(INFO) << "WrapInputLayer done in " << timer.ElapsedMSec() << "ms";

  timer.Start();
  Preprocess(img, input_channels);
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

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
template <typename DType>
void CaffeV1Classifier<DType>::WrapInputLayer(std::vector<DType *> &input_channels) {
  caffe::Blob<DType>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  DType* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    input_channels.push_back(input_data);
    input_data += width * height;
  }
}

template <typename DType>
void CaffeV1Classifier<DType>::Preprocess(const cv::Mat& img,
                            const std::vector<DType *> &input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  std::vector<cv::Mat> split_channels;
  split_channels.resize(num_channels_);
  cv::split(sample_normalized, split_channels);

  // Convert fp32 to fp16, and fill in input channels one by one
  for (int i = 0; i < num_channels_; i++) {
    memcpy(input_channels[i], split_channels[i].data, input_geometry_.area() * sizeof(DType));
  }

  CHECK(reinterpret_cast<DType *>(input_channels[0])
    == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}

template class CaffeV1Classifier<float>;
