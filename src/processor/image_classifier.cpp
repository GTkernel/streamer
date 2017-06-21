
#include "image_classifier.h"

#include "model/model_manager.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

ImageClassifier::ImageClassifier(const ModelDesc& model_desc,
                                 const Shape& input_shape, size_t num_labels)
    : NeuralNetConsumer(PROCESSOR_TYPE_IMAGE_CLASSIFIER, model_desc,
                        input_shape, {}, {SOURCE_NAME}, {SINK_NAME}),
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

ImageClassifier::ImageClassifier(const ModelDesc& model_desc, size_t num_labels)
    : NeuralNetConsumer(PROCESSOR_TYPE_IMAGE_CLASSIFIER, {SOURCE_NAME},
                        {SINK_NAME}),
      num_labels_(num_labels),
      labels_(LoadLabels(model_desc)) {}

std::shared_ptr<ImageClassifier> ImageClassifier::Create(
    const FactoryParamsType& params) {
  ModelManager& model_manager = ModelManager::GetInstance();
  std::string model_name = params.at("model");
  CHECK(model_manager.HasModel(model_name));
  ModelDesc model_desc = model_manager.GetModelDesc(model_name);
  size_t num_labels = StringToSizet(params.at("num_labels"));

  auto num_channels_pair = params.find("num_channels");
  if (num_channels_pair == params.end()) {
    return std::make_shared<ImageClassifier>(model_desc, num_labels);
  } else {
    // If num_channels is specified, then we need to use the constructor that
    // creates a hidden NeuralNetEvaluator.
    size_t num_channels = StringToSizet(num_channels_pair->second);
    Shape input_shape = Shape(num_channels, model_desc.GetInputWidth(),
                              model_desc.GetInputHeight());
    return std::make_shared<ImageClassifier>(model_desc, input_shape,
                                             num_labels);
  }
}

bool ImageClassifier::Init() { return NeuralNetConsumer::Init(); }

void ImageClassifier::Process() {
  auto layer_frame = GetFrame(SOURCE_NAME);

  // Assign labels.
  std::vector<Prediction> predictions;
  cv::Mat output = layer_frame->GetActivations();
  float* scores;
  // Currently we only support contiguously allocated cv::Mat. Considering this
  // cv::Mat should be small (e.g. 1x1000), it is most likely contiguous.
  if (output.isContinuous()) {
    scores = (float*)(output.data);
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
  for (const auto& pred : predictions) {
    tags.push_back(pred.first);
  }
  cv::Mat original_image = layer_frame->GetOriginalImage();
  Frame* metadata_frame = new Frame();
  metadata_frame->SetTags(tags);
  metadata_frame->SetOriginalImage(original_image);
  PushFrame(SINK_NAME, metadata_frame);
}

std::vector<std::string> ImageClassifier::LoadLabels(
    const ModelDesc& model_desc) {
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
