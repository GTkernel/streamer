//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "image_transformer.h"

#include "model/model_manager.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

ImageTransformer::ImageTransformer(const Shape& target_shape, bool crop,
                                   bool convert)
    : Processor(PROCESSOR_TYPE_IMAGE_TRANSFORMER, {SOURCE_NAME}, {SINK_NAME}),
      target_shape_(target_shape),
      crop_(crop),
      convert_(convert) {
  if (crop_) {
    LOG(WARNING) << "`crop` parameter is unsupported...disabling.";
    crop_ = false;
  }
  if (convert_) {
    LOG(WARNING) << "`convert` parameter is unsupported...disabling.";
    convert_ = false;
  }
}

std::shared_ptr<ImageTransformer> ImageTransformer::Create(
    const FactoryParamsType& params) {
  int width = atoi(params.at("width").c_str());
  int height = atoi(params.at("height").c_str());

  // Default channel = 3
  int num_channels = 3;
  if (params.count("channels") != 0) {
    num_channels = atoi(params.at("channels").c_str());
  }
  CHECK(width >= 0 && height >= 0 && num_channels >= 0)
      << "Width (" << width << "), height (" << height
      << "), and number of channels (" << num_channels
      << ") must not be negative.";

  return std::make_shared<ImageTransformer>(Shape(num_channels, width, height),
                                            true, true);
}

void ImageTransformer::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr ImageTransformer::GetSink() { return Processor::GetSink(SINK_NAME); }

void ImageTransformer::Process() {
  auto frame = GetFrame(SOURCE_NAME);
  auto start_time = boost::posix_time::microsec_clock::local_time();
  const cv::Mat& img = frame->GetValue<cv::Mat>("original_image");

  int num_channel = target_shape_.channel;
  int width = target_shape_.width;
  int height = target_shape_.height;
  cv::Size input_geometry(width, height);

  cv::Mat sample_image;
  // Convert channels
  if (img.channels() == 3 && num_channel == 1)
    cv::cvtColor(img, sample_image, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channel == 1)
    cv::cvtColor(img, sample_image, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channel == 3)
    cv::cvtColor(img, sample_image, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channel == 3)
    cv::cvtColor(img, sample_image, cv::COLOR_GRAY2BGR);
  else
    sample_image = img;

  cv::Mat sample_cropped;
  // Crop according to scale
  if (crop_) {
    int desired_width = (int)((float)width / height * img.size[1]);
    int desired_height = (int)((float)height / width * img.size[0]);
    int new_width = img.size[0];
    int new_height = img.size[1];
    if (desired_width < img.size[0]) {
      new_width = desired_width;
    } else {
      new_height = desired_height;
    }
    cv::Rect roi((img.size[1] - new_height) / 2, (img.size[0] - new_width) / 2,
                 new_width, new_height);
    sample_cropped = sample_image(roi);
  } else {
    sample_cropped = sample_image;
  }

  // Resize
  cv::Mat sample_resized;
  if (sample_cropped.size() != input_geometry) {
    cv::resize(sample_cropped, sample_resized, input_geometry);
  } else {
    sample_resized = sample_cropped;
  }

  // Convert to float
  cv::Mat sample_float;
  if (convert_) {
    if (num_channel == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);
  } else {
    sample_float = sample_resized;
  }
  auto end_time = boost::posix_time::microsec_clock::local_time();
  long time_elapsed = (end_time - start_time).total_microseconds();

  frame->SetValue("image", sample_float);
  frame->SetValue("image_transformer.micros", time_elapsed);
  frame->SetValue("image_transformer.enter_time", start_time);
  frame->SetValue("image_transformer.exit_time", end_time);
  PushFrame(SINK_NAME, std::move(frame));
}

bool ImageTransformer::Init() { return true; }
bool ImageTransformer::OnStop() { return true; }
