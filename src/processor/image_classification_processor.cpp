//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#include "image_classification_processor.h"
#include "model/model_manager.h"
#include "utils/utils.h"

ImageClassificationProcessor::ImageClassificationProcessor(
    std::shared_ptr<Stream> input_stream, std::shared_ptr<Stream> img_stream,
    const ModelDesc &model_desc, Shape input_shape)
    : model_desc_(model_desc), input_shape_(input_shape) {
  sources_.push_back(input_stream);
  sources_.push_back(img_stream);
  sinks_.emplace_back(new Stream);
}

bool ImageClassificationProcessor::Init() {
  // Load labels.
  CHECK(model_desc_.GetLabelFilePath() != "")
      << "Model " << model_desc_.GetName() << " has an empty label file";
  std::ifstream labels(model_desc_.GetLabelFilePath());
  CHECK(labels) << "Unable to open labels file "
                << model_desc_.GetLabelFilePath();
  string line;
  while (std::getline(labels, line)) labels_.push_back(string(line));

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

  LOG(INFO) << "Classifier initialized";
  return true;
}

bool ImageClassificationProcessor::OnStop() {
  model_.reset(nullptr);
  return true;
}

void ImageClassificationProcessor::Process() {
  Timer timer;
  auto input_stream = sources_[0];
  cv::Mat frame = input_stream->PopFrame();
  cv::Mat img = sources_[1]->PopFrame();
  CHECK(frame.channels() == input_shape_.channel &&
        frame.size[0] == input_shape_.width &&
        frame.size[1] == input_shape_.height);
  timer.Start();
  std::vector<cv::Mat> output_channels;
  float *data = (float *)input_buffer_.GetBuffer();
  for (int i = 0; i < input_shape_.channel; i++) {
    cv::Mat channel(input_shape_.height, input_shape_.width, CV_32FC1, data);
    output_channels.push_back(channel);
    data += input_shape_.width * input_shape_.height;
  }
  cv::split(frame, output_channels);
  auto predictions = Classify(1);
  //  LOG(INFO) << "Classify done in " << timer.ElapsedMSec() << " ms";

  int font_size = 1.3 * img.size[0] / 320;
  cv::putText(img, predictions[0].first, cv::Point(30, 30),
              CV_FONT_HERSHEY_COMPLEX, font_size, cvScalar(0, 0, 0), 5, CV_AA);
  cv::putText(img, predictions[0].first, cv::Point(30, 30),
              CV_FONT_HERSHEY_COMPLEX, font_size, cvScalar(200, 200, 250), 3, CV_AA);

  sinks_[0]->PushFrame(img);

  for (auto prediction : predictions) {
    LOG(INFO) << prediction.first << " " << prediction.second;
  }
}

std::vector<Prediction> ImageClassificationProcessor::Classify(int N) {
  Timer total_timer;
  Timer timer;

  total_timer.Start();
  timer.Start();

  auto input_buffer = model_->GetInputBuffer();
  input_buffer.Clone(input_buffer_);

  model_->Evaluate();
  CHECK(model_->GetOutputBuffers().size() == 1 &&
        model_->GetOutputShapes().size() == model_->GetOutputBuffers().size())
      << "Classify model does not have one buffer";
  float *scores = (float *)model_->GetOutputBuffers()[0].GetBuffer();
  LOG(INFO) << "Predict done in " << timer.ElapsedMSec() << " ms";

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN =
      Argmax(scores, model_->GetOutputShapes()[0].GetSize(), N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], scores[idx]));
  }

  LOG(INFO) << "Whole classify done in " << total_timer.ElapsedMSec() << " ms";
  return predictions;
}
