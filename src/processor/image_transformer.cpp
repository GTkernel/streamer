//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "image_transformer.h"
#include <streamer.h>

ImageTransformer::ImageTransformer(const Shape &target_shape,
                                   CropType crop_type, bool subtract_mean)
    : Processor({"input"}, {"output"}),
      target_shape_(target_shape),
      crop_type_(crop_type),
      subtract_mean_(subtract_mean) {
  auto mean_colors = ModelManager::GetInstance().GetMeanColors();
  mean_image_ =
      cv::Mat(cv::Size(target_shape.width, target_shape.height),
              target_shape.channel == 3 ? CV_32FC3 : CV_32FC1,
              cv::Scalar(mean_colors[0], mean_colors[1], mean_colors[2]));
}

void ImageTransformer::Process() {
  Timer timer;
  auto frame = GetFrame<ImageFrame>("input");
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

  if (crop_type_ == CROP_TYPE_CENTER) {
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
  } else {
    sample_cropped_ = sample_image_;
  }

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
  if (subtract_mean_) {
    cv::subtract(sample_float_, mean_image_, sample_normalized_);
    PushFrame("output",
          new ImageFrame(sample_normalized_, frame->GetOriginalImage()));
  } else {
    PushFrame("output",
          new ImageFrame(sample_float_, frame->GetOriginalImage()));
  }
}

bool ImageTransformer::Init() { return true; }
bool ImageTransformer::OnStop() { return true; }

ProcessorType ImageTransformer::GetType() {
  return PROCESSOR_TYPE_IMAGE_TRANSFORMER;
}
