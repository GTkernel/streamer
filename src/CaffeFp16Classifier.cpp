//
// Created by Ran Xian on 7/23/16.
//

#include "CaffeFp16Classifier.h"

static void get_gpus(std::vector<int>* gpus) {
  int count = 0;
#ifdef ON_TEGRA
  CUDA_CHECK(cudaGetDeviceCount(&count));
#else
  NO_GPU;
#endif
  for (int i = 0; i < count; ++i) {
    gpus->push_back(i);
  }
}

CaffeFp16Classifier::CaffeFp16Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifndef ON_TEGRA
  CHECK(false) << "Fp16 classifier can't be used in non-Tegra device";
  return;
#else
  std::vector<int> gpus;
  get_gpus(&gpus);

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
  net_.reset(new caffe::Net<float16, CAFFE_FP16_MTYPE>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  caffe::Blob<float16, CAFFE_FP16_MTYPE>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  LOG(INFO) << "Caffe net input geometry is: " << input_geometry_;
  LOG(INFO) << "Caffe net input blob shape: " << input_layer->shape_string();
  input_layer->shape_string();

  // Load the binaryproto mean file.
  SetMean(mean_file);

  // Load labels.
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  caffe::Blob<float16, CAFFE_FP16_MTYPE>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";

  // Adjust input dimensions
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  // Forward dimension change to all layers.
  net_->Reshape();
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<CaffeFp16Classifier::Prediction> CaffeFp16Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<CaffeFp16Classifier::Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void CaffeFp16Classifier::SetMean(const string& mean_file) {
  caffe::BlobProto blob_proto;
  caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  caffe::Blob<float, float> mean_blob;

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

std::vector<float> CaffeFp16Classifier::Predict(const cv::Mat& img) {
  Timer timerTotal;
  timerTotal.Start();

  Timer timer;
  std::vector<float16 *> input_channels;

  timer.Start();
  WrapInputLayer(&input_channels);
  timer.Stop();
  LOG(INFO) << "WrapInputLayer done in " << timer.ElaspedMsec() << "ms";

  timer.Start();
  Preprocess(img, &input_channels);
  timer.Stop();
  LOG(INFO) << "Preprocessing done in " << timer.ElaspedMsec() << "ms";

  timer.Start();
  net_->ForwardPrefilled();
  timer.Stop();
  LOG(INFO) << "Forward done in " << timer.ElaspedMsec() << "ms";

  /* Copy the output layer to a std::vector */
  timer.Start();
  caffe::Blob<float16, CAFFE_FP16_MTYPE>* output_layer = net_->output_blobs()[0];
  const float16* begin = output_layer->cpu_data();
  const float16* end = begin + output_layer->channels();
  timer.Stop();
  timerTotal.Stop();

  std::vector<float> scores;
  for (const float16* ptr = begin; ptr != end; ptr++) {
    scores.push_back(caffe::Get<float>(*ptr));
  }
  LOG(INFO) << "Copied output layer in " << timer.ElaspedMsec() << "ms";
  LOG(INFO) << "Whole predict done in " << timerTotal.ElaspedMsec() << "ms";
  return scores;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CaffeFp16Classifier::WrapInputLayer(std::vector<float16 *>* input_channels) {
  caffe::Blob<float16, CAFFE_FP16_MTYPE>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float16* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    input_channels->push_back(input_data);
    input_data += width * height;
  }
}

/**
 * Convert float data to half precision float
 * @param src The host source of float data to be converted.
 * @param dst The host destination to store the converted half precision float.
 */
static void float2half(float *src, float16 *dst, int n) {
  for (int i = 0; i < n; i++) {
    dst[i] = caffe::Get<float16>(src[i]);
  }
}

void CaffeFp16Classifier::Preprocess(const cv::Mat& img,
                            std::vector<float16 *>* input_channels) {
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
    sample_resized.convertTo(sample_float,  CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  cv::Mat sample_normalized_fp16;
  if (num_channels_ == 3)
    sample_normalized.convertTo(sample_normalized_fp16, CV_16UC3);
  else
    sample_normalized.convertTo(sample_normalized_fp16, CV_16UC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  std::vector<cv::Mat> split_channels;
  split_channels.resize(num_channels_);
  cv::split(sample_normalized, split_channels);

  // Convert fp32 to fp16, and fill in input channels one by one
  for (int i = 0; i < num_channels_; i++) {
    float2half((float *) split_channels[i].data, input_channels->at(i), input_geometry_.width * input_geometry_.height);
  }

  CHECK(reinterpret_cast<float16*>(input_channels->at(0))
        == net_->input_blobs()[0]->cpu_data())
  << "Input channels are not wrapping the input layer of the network.";
}

cv::Size CaffeFp16Classifier::GetInputGeometry() {
  return input_geometry_;
}