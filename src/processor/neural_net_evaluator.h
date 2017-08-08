
#ifndef STREAMER_PROCESSOR_NEURAL_NET_EVALUATOR_H_
#define STREAMER_PROCESSOR_NEURAL_NET_EVALUATOR_H_

#include <unordered_map>

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"
#include "stream/frame.h"

// A NeuralNetEvaluator is a Processor that runs deep neural network inference.
// This Processor has only one source, named "input". On creation, the
// higher-level code specifies which layers of the DNN should be published. One
// sink is created for each published layer and is named after the layer.
// At any time, PublishLayer() can be called to expose a previously unpublished
// layer.
class NeuralNetEvaluator : public Processor {
 public:
  // If output_layer_names is empty, then by default the last layer is
  // published.
  NeuralNetEvaluator(const ModelDesc& model_desc, const Shape& input_shape,
                     const std::vector<std::string>& output_layer_names = {});
  ~NeuralNetEvaluator();

  // Adds layer_name to the list of the layers whose activations will be
  // published.
  void PublishLayer(std::string layer_name);
  // Returns a vector of the names of this NeuralNetEvaluator's sinks, which are
  // the names of the layers that it is publishing.
  const std::vector<std::string> GetSinkNames() const;

  static std::shared_ptr<NeuralNetEvaluator> Create(
      const FactoryParamsType& params);

  // Hides Processor::SetSource(const std::string&, StreamPtr)
  void SetSource(const std::string& name, StreamPtr stream,
                 const std::string& layername = "");
  void SetSource(StreamPtr stream, const std::string& layername = "");

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // Executes the neural network and returns a mapping from the name of a layer
  // to that layer's activations.
  std::unordered_map<std::string, cv::Mat> Evaluate();

  Shape input_shape_;
  std::string input_layer_name_;
  std::unique_ptr<Model> model_;
};

#endif  // STREAMER_PROCESSOR_NEURAL_NET_EVALUATOR_H_
