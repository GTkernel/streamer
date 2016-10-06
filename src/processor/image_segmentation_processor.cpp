#include "image_segmentation_processor.h"
#include "model/model_manager.h"

ImageSegmentationProcessor::ImageSegmentationProcessor(
    std::shared_ptr<Stream> input_stream, std::shared_ptr<Stream> img_stream,
    const ModelDesc &model_desc, Shape input_shape)
    : model_desc_(model_desc), input_shape_(input_shape) {
  sources_.push_back(input_stream);
  sources_.push_back(img_stream);
  sinks_.emplace_back(new Stream);  // Original video frame
  sinks_.emplace_back(new Stream);  // Segmented video frame
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
  auto input_stream = sources_[0];
  cv::Mat frame = input_stream->PopFrame();
  cv::Mat original_img = sources_[1]->PopFrame();
  CHECK(frame.channels() == input_shape_.channel &&
        frame.size[0] == input_shape_.width &&
        frame.size[1] == input_shape_.height);

  // Get bytes to feed into the model
  std::vector<cv::Mat> output_channels;
  float *data = (float *)(model_->GetInputBuffer().GetBuffer());
  for (int i = 0; i < input_shape_.channel; i++) {
    cv::Mat channel(input_shape_.height, input_shape_.width, CV_32FC1, data);
    output_channels.push_back(channel);
    data += input_shape_.width * input_shape_.height;
  }
  cv::split(frame, output_channels);

  model_->Evaluate();

  DataBuffer output_buffer = model_->GetOutputBuffers()[0];
  Shape output_shape = model_->GetOutputShapes()[0];
  LOG(INFO) << output_shape.channel << " " << output_shape.width << " "
            << output_shape.height;

  // Render the segmentation
  cv::Mat wrapper(input_shape_.height, input_shape_.width, CV_32FC1,
                  output_buffer.GetBuffer());

  cv::Mat output_frame;
  wrapper.convertTo(output_frame, CV_8U, 255.0 / 21);
  LOG(INFO) << output_frame.row(10).col(10);

  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;

  cv::minMaxLoc(wrapper, &minVal, &maxVal, &minLoc, &maxLoc);

  LOG(INFO) << "min val : " << minVal;
  LOG(INFO) << "max val: " << maxVal;

  auto output_stream = sinks_[0];
  auto original_img_stream = sinks_[1];
  output_stream->PushFrame(output_frame);
  original_img_stream->PushFrame(original_img);
}
