//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "image_transformer.h"
#include <streamer.h>

ImageTransformer::ImageTransformer(std::shared_ptr<Stream> input_stream,
                                   const Shape &target_shape,
                                   CropType crop_type, bool subtract_mean)
    : Processor({input_stream}, 1),
      target_shape_(target_shape),
      crop_type_(crop_type),
      subtract_mean_(subtract_mean) {
  auto mean_colors = ModelManager::GetInstance().GetMeanColors();
  mean_image_ =
      cv::Mat(cv::Size(target_shape.width, target_shape.height), CV_32FC3,
              cv::Scalar(mean_colors[0], mean_colors[1], mean_colors[2]));
}

void ImageTransformer::Process() {
  Timer timer;
  auto frame = GetFrame<ImageFrame>(0);
  cv::Mat img = frame->GetImage();
  timer.Start();

  int num_channel = target_shape_.channel, width = target_shape_.width,
      height = target_shape_.height;
  cv::Size input_geometry(width, height);

  // Convert channels
  if (img.channels() == 3 && num_channel == 1)
    cv::cvtColor(img, sample_image_, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channel == 1)
    cv::cvtColor(img, sample_image_, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channel == 3)
    cv::cvtColor(img, sample_image_, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channel == 3)
    cv::cvtColor(img, sample_image_, cv::COLOR_GRAY2BGR);
  else
    sample_image_ = img;

  // Crop according to scale
  int desired_width = (int)((float)width / height * img.size[1]);
  int desired_height = (int)((float)height / width * img.size[0]);
  int new_width = img.size[0], new_height = img.size[1];
  if (desired_width < img.size[0]) {
    new_width = desired_width;
  } else {
    new_height = desired_height;
  }
  cv::Rect roi((img.size[1] - new_height) / 2, (img.size[0] - new_width) / 2,
               new_width, new_height);
  sample_cropped_ = sample_image_(roi);

  // Resize
  if (sample_cropped_.size() != input_geometry)
    cv::resize(sample_cropped_, sample_resized_, input_geometry);
  else
    sample_resized_ = sample_cropped_;

  // Convert to float
  if (num_channel == 3)
    sample_resized_.convertTo(sample_float_, CV_32FC3);
  else
    sample_resized_.convertTo(sample_float_, CV_32FC1);

  // Normalize
  cv::subtract(sample_float_, mean_image_, sample_normalized_);

  PushFrame(0, new ImageFrame(sample_normalized_, frame->GetOriginalImage()));
}

bool ImageTransformer::Init() { return true; }
bool ImageTransformer::OnStop() { return true; }