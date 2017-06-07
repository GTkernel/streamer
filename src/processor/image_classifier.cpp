
#include "processor/image_classifier.h"
#include "utils/math_utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

ImageClassifier::ImageClassifier(const ModelDesc &model_desc,
                                 const Shape &input_shape, size_t num_labels)
    : NeuralNetConsumer(model_desc, input_shape, {}, {SOURCE_NAME},
                        {SINK_NAME}),
      num_labels_(num_labels),
      labels_(LoadLabels(model_desc)) {
  // The NeuralNetEvaluator only has one sink, but we don't know what it's
  // called because it's named after the last layer in the model, which could
  // be named anything.
  std::string nne_sink_name = nne_->GetSinkNames().at(0);
  StreamPtr stream = nne_->GetSink(nne_sink_name);
  // Call Processor::SetSource() because NeuralNetConsumer::SetSource() would
  // set the NeuralNetEvaluator's source (because the NeuralNetEvaluator is
  // private).
  Processor::SetSource(SOURCE_NAME, stream);
}

ImageClassifier::ImageClassifier(const ModelDesc &model_desc, size_t num_labels)
    : NeuralNetConsumer({SOURCE_NAME}, {SINK_NAME}),
      num_labels_(num_labels),
      labels_(LoadLabels(model_desc)) {}

ProcessorType ImageClassifier::GetType() const {
  return PROCESSOR_TYPE_IMAGE_CLASSIFIER;
}

bool ImageClassifier::Init() { return NeuralNetConsumer::Init(); }

void ImageClassifier::Process() {
  std::shared_ptr<LayerFrame> layer_frame = GetFrame<LayerFrame>(SOURCE_NAME);

  // Assign labels.
  std::vector<Prediction> predictions;
  cv::Mat output = layer_frame->GetActivations();
  float *scores;
  // Currently we only support contiguously allocated cv::Mat. Considering this
  // cv::Mat should be small (e.g. 1x1000), it is most likely contiguous.
  if (output.isContinuous()) {
    scores = (float *)(output.data);
  } else {
    LOG(FATAL)
        << "Non-contiguous allocation of cv::Mat is currently not supported";
  }
  // using labels_.size() completely defeats the purpose and also causes issues
  // elsewhere
  std::vector<int> top_label_idxs = Argmax(scores, output.size[1], num_labels_);
  for (decltype(num_labels_) i = 0; i < num_labels_; ++i) {
    int label_idx = top_label_idxs.at(i);
    predictions.push_back(
        std::make_pair(labels_.at(label_idx), scores[label_idx]));
  }

  // Create and push a MetadataFrame.
  std::vector<std::string> tags;
  for (const auto &pred : predictions) {
    tags.push_back(pred.first);
  }
  cv::Mat original_image = layer_frame->GetOriginalImage();
  PushFrame(SINK_NAME, new MetadataFrame(tags, original_image,
                                         layer_frame->GetStartTime()));
}

std::vector<std::string> ImageClassifier::LoadLabels(
    const ModelDesc &model_desc) {
  // Load labels
  std::string labels_filepath = model_desc.GetLabelFilePath();
  CHECK(labels_filepath != "") << "Empty label file: " << labels_filepath;
  std::ifstream labels_stream(labels_filepath);
  CHECK(labels_stream) << "Unable to open labels file: " << labels_filepath;

  std::string line;
  std::vector<std::string> labels;
  while (std::getline(labels_stream, line)) {
    labels.push_back(std::string(line));
  }
  return labels;
}
