//
// Created by Ran Xian (xranthoar@gmail.com) on 10/6/16.
//

#include "image_classification_processor.h"
#include "model/model_manager.h"
#include "utils/utils.h"

ImageClassificationProcessor::ImageClassificationProcessor(
    std::vector<std::shared_ptr<Stream>> input_streams,
    std::vector<std::shared_ptr<Stream>> img_streams,
    const ModelDesc &model_desc, Shape input_shape)
    : model_desc_(model_desc), input_shape_(input_shape) {
  CHECK(input_streams.size() == img_streams.size());
  batch_size_ = input_streams.size();
  LOG(INFO) << "batch size of " << batch_size_;
  for (auto stream : input_streams) {
    sources_.push_back(stream);
  }
  for (auto stream : img_streams) {
    sources_.push_back(stream);
  }
  for (int i = 0; i < img_streams.size(); i++) {
    sinks_.emplace_back(new Stream);
  }
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
  model_ = manager.CreateModel(model_desc_, input_shape_, sinks_.size());
  model_->Load();

  // Create mean image
  auto mean_colors = manager.GetMeanColors();
  mean_image_ =
      cv::Mat(cv::Size(input_shape_.width, input_shape_.height), CV_32FC3,
              cv::Scalar(mean_colors[0], mean_colors[1], mean_colors[2]));

  // Prepare data buffer
  input_buffer_ =
      DataBuffer(batch_size_ * input_shape_.GetSize() * sizeof(float));

  LOG(INFO) << "Classifier initialized";
  return true;
}

bool ImageClassificationProcessor::OnStop() {
  model_.reset(nullptr);
  return true;
}

void ImageClassificationProcessor::Process() {
  Timer timer;
  float *data = (float *)input_buffer_.GetBuffer();
  for (int i = 0; i < batch_size_; i++) {
    auto input_stream = sources_[i];
    cv::Mat frame = input_stream->PopFrame().GetImage();
    CHECK(frame.channels() == input_shape_.channel &&
          frame.size[0] == input_shape_.width &&
          frame.size[1] == input_shape_.height);
    std::vector<cv::Mat> output_channels;
    for (int i = 0; i < input_shape_.channel; i++) {
      cv::Mat channel(input_shape_.height, input_shape_.width, CV_32FC1, data);
      output_channels.push_back(channel);
      data += input_shape_.width * input_shape_.height;
    }
    cv::split(frame, output_channels);
  }

  timer.Start();
  auto predictions = Classify(1);
  double fps = 1000.0 / timer.ElapsedMSec();

  for (int i = 0; i < batch_size_; i++) {
    cv::Mat img = sources_[batch_size_ + i]->PopFrame().GetImage();
    double font_size = 0.8 * img.size[0] / 320.0;
     cv::putText(img, predictions[i][0].first.substr(10), cv::Point(img.rows / 3, img.cols / 3),
                CV_FONT_HERSHEY_DUPLEX, font_size, cvScalar(0, 0, 0), 12,
                CV_AA);
    cv::putText(img, predictions[i][0].first.substr(10), cv::Point(img.rows / 3, img.cols / 3),
                CV_FONT_HERSHEY_DUPLEX, font_size, cvScalar(200, 200, 250), 2,
                CV_AA);

    char fps_string[256];
    sprintf(fps_string, "%.2lffps", fps);
    cv::putText(img, fps_string, cv::Point(img.rows / 3, img.cols / 6),
                CV_FONT_HERSHEY_DUPLEX, font_size, cvScalar(0,0,0), 12,
                CV_AA);
    cv::putText(img, fps_string, cv::Point(img.rows / 3, img.cols / 6),
                CV_FONT_HERSHEY_DUPLEX, font_size, cvScalar(200, 200, 250), 2,
                CV_AA);
    sinks_[i]->PushFrame(img);
    for (auto prediction : predictions[i]) {
      LOG(INFO) << prediction.first << " " << prediction.second;
    }
  }
}

std::vector<std::vector<Prediction>> ImageClassificationProcessor::Classify(int N) {
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
  N = std::min<int>(labels_.size(), N);
  std::vector<std::vector<Prediction>> results;
  for (int i = 0; i < batch_size_; i++) {
    std::vector<int> maxN =
        Argmax(scores, model_->GetOutputShapes()[0].channel, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
      int idx = maxN[i];
      predictions.push_back(std::make_pair(labels_[idx], scores[idx]));
    }
    results.push_back(predictions);
    scores += model_->GetOutputShapes()[0].channel;
  }

  LOG(INFO) << "Whole classify done in " << total_timer.ElapsedMSec() << " ms";
  return results;
}
