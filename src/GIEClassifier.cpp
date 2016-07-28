//
// Created by Ran Xian on 7/27/16.
//

#include "GIEClassifier.h"
#include <cuda_fp16.h>
#include "fp16_emu.h"
#include <fstream>

half cpu_float2half_rn(float f)
{
  half ret;

  unsigned x = *((int*)(void*)(&f));
  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    ret.x = 0x7fffU;
    return ret;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    ret.x = sign | 0x7c00U;
    return ret;
  }
  if (u < 0x33000001) {
    ret.x = (sign | 0x0000);
    return ret;
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  ret.x = (sign | (exponent << 10) | mantissa);

  return ret;
}

float cpu_half2float(half h)
{
  unsigned sign = ((h.x >> 15) & 1);
  unsigned exponent = ((h.x >> 10) & 0x1f);
  unsigned mantissa = ((h.x & 0x3ff) << 13);

  if (exponent == 0x1f) {  /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) {  /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1;  /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }

  int temp = ((sign << 31) | (exponent << 23) | mantissa);

  return *((float*)((void*)&temp));
}

GIEClassifier::GIEClassifier(const string &deploy_file,
                             const string &model_file,
                             const string &mean_file,
                             const string &label_file): inferer_(deploy_file, model_file, "data", "prob")
{
  inferer_.CreateEngine();
  // Set dimensions
  input_geometry_ = cv::Size(inferer_.GetInputShape().width, inferer_.GetInputShape().height);
  num_channels_ = inferer_.GetInputShape().channel;
  // Load the binaryproto mean file.
  SetMean(mean_file);

  // Load labels.
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  // Allocate input data and output data
  input_data_ = new float16[inferer_.GetInputShape().Volumn()];
  output_data_ = new float16[inferer_.GetOutputShape().Volumn()];
}

GIEClassifier::~GIEClassifier() {
  delete[] input_data_;
  delete[] output_data_;
  inferer_.DestroyEngine();
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/**
 * Convert float data to half precision float
 * @param src The host source of float data to be converted.
 * @param dst The host destination to store the converted half precision float.
 */
static void float2half(float *src, float16 *dst, int n) {
  LOG(INFO) << "Copy from float:" << std::hex << src << "to float16:" << std::hex << dst << " of " << n << " elements";
  for (int i = 0; i < n; i++) {
    dst[i] = cpu_float2half_rn(src[i]);
  }
}

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

std::vector<GIEClassifier::Prediction> GIEClassifier::Classify(const cv::Mat &img, int N) {
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

/**
 * There is some code duplication, need refactor.
 */
void GIEClassifier::SetMean(const string &mean_file) {
  IBinaryProtoBlob *meanBlob = CaffeParser::parseBinaryProto(mean_file.c_str());
  const float *data = reinterpret_cast<const float *>(meanBlob->getData());
  float *mutable_data = new float[input_geometry_.height * input_geometry_.width * num_channels_];
  memcpy(mutable_data, data, input_geometry_.height * input_geometry_.width * num_channels_);
  float *mutable_data_ptr = mutable_data;
  std::vector<cv::Mat> channels;
  LOG(INFO) << input_geometry_;
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, mutable_data_ptr);
    channels.push_back(channel);
    mutable_data_ptr += input_geometry_.height * input_geometry_.width;
  }
  meanBlob->destroy();
  delete[] mutable_data;

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> GIEClassifier::Predict(const cv::Mat &img) {
  Timer micro_timer;
  Timer macro_timer;
  macro_timer.Start();
  micro_timer.Start();
  CreateInput(img);
  micro_timer.Stop();
  LOG(INFO) << "CreateInput took " << micro_timer.ElaspedMsec() << " ms";
  micro_timer.Start();
  inferer_.DoInference(input_data_, output_data_);
  micro_timer.Stop();
  LOG(INFO) << "DoInference took " << micro_timer.ElaspedMsec() << " ms";
  int output_channels = inferer_.GetOutputShape().channel;
  std::vector<float> scores;
  micro_timer.Start();
  LOG(INFO) << "Output channel is " << output_channels;
  for (int i = 0; i < output_channels; i++) {
    scores.push_back(cpu_half2float(output_data_[i]));
  }
  micro_timer.Stop();
  LOG(INFO) << "Copy output took " << micro_timer.ElaspedMsec() << " ms";
  macro_timer.Stop();
  LOG(INFO) << "Whole prediction done in " << macro_timer.ElaspedMsec() << " ms";
  return scores;
}

void GIEClassifier::CreateInput(const cv::Mat &img) {
  Shape input_shape = inferer_.GetInputShape();
  int channel = input_shape.channel;
  int width = input_shape.width;
  int height = input_shape.height;
  CHECK_EQ(img.channels(), channel) << "Input image channel must equal to network channel";
  CHECK(channel == 3) << "Can only deal with 3-channel BGR images now";
  cv::Mat resized_sample;
  if (img.size[0] != width || img.size[1] == height) {
    cv::resize(img, resized_sample, cv::Size(width, height));
  } else {
    resized_sample = img;
  }

  cv::Mat sample_float;
  resized_sample.convertTo(sample_float, CV_32FC3);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  std::vector<cv::Mat> split_channels(num_channels_);
  cv::split(sample_normalized, split_channels);

  CHECK(split_channels.size() == num_channels_);

  float16 *input_pointer = input_data_;
  for (int i = 0; i < num_channels_; i++) {
    float2half((float *)split_channels[i].data, input_pointer, width * height);
    input_pointer += width * height;
  }
}