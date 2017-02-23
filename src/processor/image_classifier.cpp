//
// Created by Ran Xian (xranthoar@gmail.com) on 10/6/16.
//

#include "image_classifier.h"
#include "model/model_manager.h"
#include "utils/utils.h"

ImageClassifier::ImageClassifier(const ModelDesc &model_desc, Shape input_shape,
                                 size_t batch_size)
    : Processor({}, {}),  // Sources and sinks will be initialized by ourselves
      model_desc_(model_desc),
      input_shape_(input_shape),
      batch_size_(batch_size) {
  for (size_t i = 0; i < batch_size; i++) {
    sources_.insert({"input" + std::to_string(i), nullptr});
    sinks_.insert(
        {"output" + std::to_string(i), std::shared_ptr<Stream>(new Stream)});
  }
  LOG(INFO) << "batch size of " << batch_size_;
}

bool ImageClassifier::Init() {
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
  model_ = manager.CreateModel(model_desc_, input_shape_, batch_size_);
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

bool ImageClassifier::OnStop() {
  model_.reset(nullptr);
  return true;
}

#define GET_SOURCE_NAME(i) ("input" + std::to_string(i))
#define GET_SINK_NAME(i) ("output" + std::to_string(i))

void ImageClassifier::Process() {
  Timer timer;
  timer.Start();

  std::vector<std::shared_ptr<ImageFrame>> image_frames;
  float *data = (float *)input_buffer_.GetBuffer();
  for (int i = 0; i < batch_size_; i++) {
    auto image_frame = GetFrame<ImageFrame>(GET_SOURCE_NAME(i));
    image_frames.push_back(image_frame);
    cv::Mat img = image_frame->GetImage();
    CHECK(img.channels() == input_shape_.channel &&
          img.size[1] == input_shape_.width &&
          img.size[0] == input_shape_.height);
    std::vector<cv::Mat> output_channels;
    for (int j = 0; j < input_shape_.channel; j++) {
      cv::Mat channel(input_shape_.height, input_shape_.width, CV_32FC1, data);
      output_channels.push_back(channel);
      data += input_shape_.width * input_shape_.height;
    }
    cv::split(img, output_channels);
  }

  auto predictions = Classify(1);

  for (int i = 0; i < batch_size_; i++) {
    auto frame = image_frames[i];
    cv::Mat img = frame->GetOriginalImage();
    CHECK(!img.empty());
    string predict_label = predictions[i][0].first;
    PushFrame(GET_SINK_NAME(i), new MetadataFrame({predict_label}, img));
    for (auto prediction : predictions[i]) {
      LOG(INFO) << prediction.first << " " << prediction.second;
    }
  }

  LOG(INFO) << "Classification took " << timer.ElapsedMSec() << " ms";
}

std::vector<std::vector<Prediction>> ImageClassifier::Classify(int N) {
  std::vector<std::vector<Prediction>> results;

  auto input_buffer = model_->GetInputBuffer();
  input_buffer.Clone(input_buffer_);

  model_->Evaluate();
  CHECK(model_->GetOutputBuffers().size() == 1 &&
        model_->GetOutputShapes().size() == model_->GetOutputBuffers().size())
      << "Classify model does not have one buffer";
  float *scores = (float *)model_->GetOutputBuffers()[0].GetBuffer();
  N = std::min<int>(labels_.size(), N);
  for (int i = 0; i < batch_size_; i++) {
    CHECK(model_->GetOutputShapes()[0].GetSize() == 1000);
    std::vector<int> maxN =
        Argmax(scores, model_->GetOutputShapes()[0].GetSize(), N);
    std::vector<Prediction> predictions;
    for (int j = 0; j < N; ++j) {
      int idx = maxN[j];
      predictions.push_back(std::make_pair(labels_[idx], scores[idx]));
    }
    results.push_back(predictions);
    scores += model_->GetOutputShapes()[0].GetSize();
  }

  return results;
}

ProcessorType ImageClassifier::GetType() {
  return PROCESSOR_TYPE_IMAGE_CLASSIFIER;
}

void ImageClassifier::SetInputStream(int src_id, StreamPtr stream) {
  SetSource(GET_SOURCE_NAME(src_id), stream);
}
