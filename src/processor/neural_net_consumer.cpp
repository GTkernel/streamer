
#include "processor/neural_net_consumer.h"
#include "utils/utils.h"

NeuralNetConsumer::NeuralNetConsumer(
    ProcessorType type, const ModelDesc& model_desc, const Shape& input_shape,
    const std::vector<std::string>& output_layer_names,
    const std::vector<std::string>& source_names,
    const std::vector<std::string>& sink_names)
    : Processor(type, source_names, sink_names),
      output_layer_names_(output_layer_names),
      nne_(new NeuralNetEvaluator(model_desc, input_shape,
                                  output_layer_names)) {}

NeuralNetConsumer::NeuralNetConsumer(
    ProcessorType type, const std::vector<std::string>& source_names,
    const std::vector<std::string>& sink_names)
    : Processor(type, source_names, sink_names) {}

void NeuralNetConsumer::SetSource(const string& name, StreamPtr stream) {
  if (NneIsPrivate()) {
    // If we are managing the NeuralNetEvaluator, then set its source instead.
    nne_->SetSource(name, stream, "");
  } else {
    Processor::SetSource(name, stream);
  }
}

double NeuralNetConsumer::GetTrailingAvgProcessingLatencyMs() const {
  double our_latency = Processor::GetTrailingAvgProcessingLatencyMs();
  if (NneIsPrivate()) {
    // We add our latency to the latency of our hidden NeuralNetEvaluator.
    return nne_->GetTrailingAvgProcessingLatencyMs() + our_latency;
  } else {
    return our_latency;
  }
}

double NeuralNetConsumer::GetAvgProcessingLatencyMs() const {
  double our_latency = Processor::GetAvgProcessingLatencyMs();
  if (NneIsPrivate()) {
    // We add our latency to the latency of our hidden NeuralNetEvaluator.
    return nne_->GetAvgProcessingLatencyMs() + our_latency;
  } else {
    return our_latency;
  }
}

bool NeuralNetConsumer::Init() {
  if (NneIsPrivate()) {
    return nne_->Start();
  } else {
    return true;
  }
}

bool NeuralNetConsumer::OnStop() {
  if (NneIsPrivate()) {
    bool result = nne_->Stop();
    delete nne_;
    return result;
  } else {
    return true;
  }
}

bool NeuralNetConsumer::NneIsPrivate() const { return nne_ != NULL; }
