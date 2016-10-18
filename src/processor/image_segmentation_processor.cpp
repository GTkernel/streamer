#include "image_segmentation_processor.h"
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "model/model_manager.h"

ImageSegmentationProcessor::ImageSegmentationProcessor(
    std::shared_ptr<Stream> input_stream, const ModelDesc &model_desc,
    Shape input_shape)
    : model_desc_(model_desc), input_shape_(input_shape) {
  sources_.push_back(input_stream);
  sinks_.emplace_back(new Stream);
}

bool ImageSegmentationProcessor::Init() {
  // Load model
  auto &manager = ModelManager::GetInstance();
  model_ = manager.CreateModel(model_desc_, input_shape_);
  model_->Load();

  // Create mean image
  auto mean_colors = manager.GetMeanColors();
  mean_image_ =
      cv::Mat(cv::Size(input_shape_.width, input_shape_.height), CV_32FC3,
              cv::Scalar(mean_colors[0], mean_colors[1], mean_colors[2]));

  // Prepare data buffer
  input_buffer_ = DataBuffer(input_shape_.GetSize() * sizeof(float));

  LOG(INFO) << "Processor initialized";
  return true;
}

bool ImageSegmentationProcessor::OnStop() {
  model_.reset(nullptr);
  return true;
}

void ImageSegmentationProcessor::Process() {
  // Do image segmentation
  Timer timer;
  timer.Start();
  auto frame = PopImageFrame(0);
  cv::Mat image = frame->GetImage();
  cv::Mat original_image = frame->GetOriginalImage();

  CHECK(image.channels() == input_shape_.channel &&
        image.size[0] == input_shape_.width &&
        image.size[1] == input_shape_.height);

  // Get bytes to feed into the model
  std::vector<cv::Mat> output_channels;
  float *data = (float *)(model_->GetInputBuffer().GetBuffer());
  for (int i = 0; i < input_shape_.channel; i++) {
    cv::Mat channel(input_shape_.height, input_shape_.width, CV_32FC1, data);
    output_channels.push_back(channel);
    data += input_shape_.width * input_shape_.height;
  }
  cv::split(image, output_channels);

  model_->Evaluate();

  DataBuffer output_buffer = model_->GetOutputBuffers()[0];
  Shape output_shape = model_->GetOutputShapes()[0];
  LOG(INFO) << output_shape.channel << " " << output_shape.width << " "
            << output_shape.height;

  // Render the segmentation
  cv::Mat output_img =
      cv::Mat::zeros(output_shape.height, output_shape.width, CV_8U);
  cv::Mat output_score =
      cv::Mat::zeros(output_shape.height, output_shape.width, CV_32F);
  float *output_data = (float *)output_buffer.GetBuffer();
  for (int i = 0; i < output_shape.channel; i++) {
    cv::Mat wrapper(input_shape_.height, input_shape_.width, CV_32FC1,
                    output_data);
    for (int j = 0; j < input_shape_.height; j++) {
      for (int k = 0; k < input_shape_.width; k++) {
        if (wrapper.at<float>(j, k) > output_score.at<float>(j, k)) {
          output_score.at<float>(j, k) = wrapper.at<float>(j, k);
          output_img.at<uchar>(j, k) = (uchar)i;
        }
      }
    }
    output_data += input_shape_.width * input_shape_.height;
  }

  cv::Mat output_frame;
  cv::Mat colored_output;
  output_img.convertTo(output_frame, CV_8U, 255.0 / 21);

  cv::applyColorMap(output_frame, colored_output, 4);
  cv::resize(colored_output, colored_output,
             cv::Size(original_image.cols, original_image.rows));

  PushFrame(0, new ImageFrame(colored_output, frame->GetOriginalImage()));
  LOG(INFO) << "Segmentation takes " << timer.ElapsedMSec() << " ms";
}
