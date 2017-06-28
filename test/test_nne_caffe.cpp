
#include <gtest/gtest.h>
#include <boost/algorithm/string/replace.hpp>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#include <assert.h>
#include <iostream>
#include <string>
#include <vector>

#include "common/types.h"
#include "model/model.h"
#include "processor/neural_net_evaluator.h"
#include "common/serialization.h"
#include "stream/frame.h"
#include "stream/stream.h"

constexpr auto ALPHA = 0.01f;

constexpr auto CHANNELS = 3;
constexpr auto WIDTH = 229;
constexpr auto HEIGHT = 229;
const auto SHAPE = cv::Size(WIDTH, HEIGHT);

constexpr auto MEAN_IMAGE_FILEPATH = "data/inception/imagenet_mean.binaryproto";
constexpr auto INPUT_IMAGE_FILEPATH = "data/input.jpg";
constexpr auto NETWORK_FILEPATH = "data/inception/deploy.prototxt";
constexpr auto WEIGHTS_FILEPATH = "/tmp/bvlc_googlenet.caffemodel";

const std::vector<std::string> OUTPUTS = {
    "conv1/7x7_s2",
    "conv1/relu_7x7",
    "conv2/3x3",
    "conv2/3x3_reduce",
    "conv2/norm2",
    "conv2/relu_3x3",
    "conv2/relu_3x3_reduce",
    "inception_3a/1x1",
    "inception_3a/3x3",
    "inception_3a/3x3_reduce",
    "inception_3a/5x5",
    "inception_3a/5x5_reduce",
    "inception_3a/output",
    "inception_3a/output_inception_3a/output_0_split",
    "inception_3a/pool",
    "inception_3a/pool_proj",
    "inception_3a/relu_1x1",
    "inception_3a/relu_3x3",
    "inception_3a/relu_3x3_reduce",
    "inception_3a/relu_5x5",
    "inception_3a/relu_5x5_reduce",
    "inception_3a/relu_pool_proj",
    "inception_3b/1x1",
    "inception_3b/3x3",
    "inception_3b/3x3_reduce",
    "inception_3b/5x5",
    "inception_3b/5x5_reduce",
    "inception_3b/output",
    "inception_3b/pool",
    "inception_3b/pool_proj",
    "inception_3b/relu_1x1",
    "inception_3b/relu_3x3",
    "inception_3b/relu_3x3_reduce",
    "inception_3b/relu_5x5",
    "inception_3b/relu_5x5_reduce",
    "inception_3b/relu_pool_proj",
    "inception_4a/1x1",
    "inception_4a/3x3",
    "inception_4a/3x3_reduce",
    "inception_4a/5x5",
    "inception_4a/5x5_reduce",
    "inception_4a/output",
    "inception_4a/output_inception_4a/output_0_split",
    "inception_4a/pool",
    "inception_4a/pool_proj",
    "inception_4a/relu_1x1",
    "inception_4a/relu_3x3",
    "inception_4a/relu_3x3_reduce",
    "inception_4a/relu_5x5",
    "inception_4a/relu_5x5_reduce",
    "inception_4a/relu_pool_proj",
    "inception_4b/1x1",
    "inception_4b/3x3",
    "inception_4b/3x3_reduce",
    "inception_4b/5x5",
    "inception_4b/5x5_reduce",
    "inception_4b/output",
    "inception_4b/output_inception_4b/output_0_split",
    "inception_4b/pool",
    "inception_4b/pool_proj",
    "inception_4b/relu_1x1",
    "inception_4b/relu_3x3",
    "inception_4b/relu_3x3_reduce",
    "inception_4b/relu_5x5",
    "inception_4b/relu_5x5_reduce",
    "inception_4b/relu_pool_proj",
    "inception_4c/1x1",
    "inception_4c/3x3",
    "inception_4c/3x3_reduce",
    "inception_4c/5x5",
    "inception_4c/5x5_reduce",
    "inception_4c/output",
    "inception_4c/output_inception_4c/output_0_split",
    "inception_4c/pool",
    "inception_4c/pool_proj",
    "inception_4c/relu_1x1",
    "inception_4c/relu_3x3",
    "inception_4c/relu_3x3_reduce",
    "inception_4c/relu_5x5",
    "inception_4c/relu_5x5_reduce",
    "inception_4c/relu_pool_proj",
    "inception_4d/1x1",
    "inception_4d/3x3",
    "inception_4d/3x3_reduce",
    "inception_4d/5x5",
    "inception_4d/5x5_reduce",
    "inception_4d/output",
    "inception_4d/output_inception_4d/output_0_split",
    "inception_4d/pool",
    "inception_4d/pool_proj",
    "inception_4d/relu_1x1",
    "inception_4d/relu_3x3",
    "inception_4d/relu_3x3_reduce",
    "inception_4d/relu_5x5",
    "inception_4d/relu_5x5_reduce",
    "inception_4d/relu_pool_proj",
    "inception_4e/1x1",
    "inception_4e/3x3",
    "inception_4e/3x3_reduce",
    "inception_4e/5x5",
    "inception_4e/5x5_reduce",
    "inception_4e/output",
    "inception_4e/pool",
    "inception_4e/pool_proj",
    "inception_4e/relu_1x1",
    "inception_4e/relu_3x3",
    "inception_4e/relu_3x3_reduce",
    "inception_4e/relu_5x5",
    "inception_4e/relu_5x5_reduce",
    "inception_4e/relu_pool_proj",
    "inception_5a/1x1",
    "inception_5a/3x3",
    "inception_5a/3x3_reduce",
    "inception_5a/5x5",
    "inception_5a/5x5_reduce",
    "inception_5a/output",
    "inception_5a/output_inception_5a/output_0_split",
    "inception_5a/pool",
    "inception_5a/pool_proj",
    "inception_5a/relu_1x1",
    "inception_5a/relu_3x3",
    "inception_5a/relu_3x3_reduce",
    "inception_5a/relu_5x5",
    "inception_5a/relu_5x5_reduce",
    "inception_5a/relu_pool_proj",
    "inception_5b/1x1",
    "inception_5b/3x3",
    "inception_5b/3x3_reduce",
    "inception_5b/5x5",
    "inception_5b/5x5_reduce",
    "inception_5b/output",
    "inception_5b/pool",
    "inception_5b/pool_proj",
    "inception_5b/relu_1x1",
    "inception_5b/relu_3x3",
    "inception_5b/relu_3x3_reduce",
    "inception_5b/relu_5x5",
    "inception_5b/relu_5x5_reduce",
    "inception_5b/relu_pool_proj",
    "loss3/classifier",
    "pool1/3x3_s2",
    "pool1/norm1",
    "pool2/3x3_s2",
    "pool2/3x3_s2_pool2/3x3_s2_0_split",
    "pool3/3x3_s2",
    "pool3/3x3_s2_pool3/3x3_s2_0_split",
    "pool4/3x3_s2",
    "pool4/3x3_s2_pool4/3x3_s2_0_split",
    "pool5/7x7_s1",
    "pool5/drop_7x7_s1",
    "prob"};

bool FloatEqual(float lhs, float rhs) {
  if (lhs < 0) {
    lhs *= -1;
    rhs *= -1;
  }
  return lhs - lhs * ALPHA <= rhs && lhs + lhs * ALPHA >= rhs;
}

void CvMatEqual(cv::Mat lhs, cv::Mat rhs) {
  std::vector<int> cur_idx;
  if (lhs.dims != rhs.dims) {
    return;
  }
  CHECK_EQ(lhs.dims, rhs.dims);
  CHECK_GT(lhs.dims, 0);
  auto lhs_it = lhs.begin<float>();
  auto lhs_end = lhs.end<float>();
  auto rhs_it = rhs.begin<float>();
  auto rhs_end = rhs.end<float>();
  while (lhs_it != lhs_end && rhs_it != rhs_end) {
    CHECK(FloatEqual(*lhs_it, *rhs_it))
        << "Expects: " << *lhs_it << " Found: " << *rhs_it;
    ++lhs_it;
    ++rhs_it;
  }
}

cv::Mat GetMean() {
  caffe::BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(MEAN_IMAGE_FILEPATH, &blob_proto);

  // Convert from BlobProto to Blob<float>
  caffe::Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);

  // The format of the mean file is planar 32-bit float BGR or grayscale.
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (unsigned int i = 0; i < 3; ++i) {
    // Extract an individual channel.
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  // Merge the separate channels into a single image.
  cv::Mat mean;
  cv::merge(channels, mean);

  // Compute the global mean pixel value and create a mean image
  // filled with this value.
  cv::Scalar channel_mean = cv::mean(mean);
  return cv::Mat(SHAPE, mean.type(), channel_mean);
}

cv::Mat Preprocess(const cv::Mat& img) {
  // Convert the input image to the input image format of the network.
  cv::Mat sample;
  sample = img;

  cv::Mat sample_resized;
  cv::resize(sample, sample_resized, SHAPE);

  cv::Mat sample_float;
  sample_resized.convertTo(sample_float, CV_32FC3);

  cv::Mat sample_normalized;
  cv::Mat mean = GetMean();
  cv::subtract(sample_float, mean, sample_normalized);

  // This operation will write the separate BGR planes directly to the
  // input layer of the network because it is wrapped by the cv::Mat
  // objects in input_channels.
  return sample_normalized;
}

TEST(TestNneCaffe, TestExtractIntermediateActivationsCaffe) {
  std::ifstream f(WEIGHTS_FILEPATH);
  ASSERT_TRUE(f.good())
      << "The Caffe model file \"" << WEIGHTS_FILEPATH
      << "\" was not found. Download it by executing: "
      << "curl -o " << WEIGHTS_FILEPATH
      << " http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel";
  f.close();

  // Construct model
  Shape input_shape(CHANNELS, WIDTH, HEIGHT);
  ModelDesc desc("TestLayerOutputModel", MODEL_TYPE_CAFFE, NETWORK_FILEPATH,
                 WEIGHTS_FILEPATH, WIDTH, HEIGHT);
  NeuralNetEvaluator nne(desc, input_shape, OUTPUTS);

  // Read the input
  cv::Mat img = cv::imread(INPUT_IMAGE_FILEPATH, -1);
  EXPECT_FALSE(img.empty());
  cv::Mat input_data = Preprocess(img);

  // Construct frame with input image in it
  std::shared_ptr<ImageFrame> input_frame =
      std::make_shared<ImageFrame>(input_data, img);

  // Prepare input stream
  StreamPtr input_stream_ptr = std::make_shared<Stream>();

  nne.SetSource("input", input_stream_ptr);

  std::vector<std::pair<std::string, StreamReader*>> readers;
  EXPECT_EQ(nne.GetSinkNames().size(), OUTPUTS.size());
  for (const auto& sink_name : nne.GetSinkNames()) {
    readers.push_back({sink_name, nne.GetSink(sink_name)->Subscribe()});
  }
  nne.Start();
  input_stream_ptr->PushFrame(input_frame);
  for (const auto& pair : readers) {
    auto sink_name = pair.first;
    auto reader = pair.second;
    auto result_frame = reader->PopFrame<LayerFrame>();
    ASSERT_FALSE(result_frame == nullptr) << "Unable to get frame";

    std::string filename = sink_name;
    boost::replace_all(filename, "/", ".");
    filename = "data/inception/caffe_ground_truth/" + filename + ".expect";
    // contains the activations of the output layer
    std::ifstream input_gt_file;
    input_gt_file.open(filename);
    cv::Mat expected_output;
    boost::archive::binary_iarchive expected_output_archive(input_gt_file);
    try {
      expected_output_archive >> expected_output;
    } catch (std::exception e) {
      LOG(INFO) << "Ignoring empty layer\n";
    }
    input_gt_file.close();
    CvMatEqual(
        expected_output,
        std::dynamic_pointer_cast<LayerFrame>(result_frame)->GetActivations());
  }

  nne.Stop();
}
