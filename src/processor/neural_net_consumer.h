
#ifndef STREAMER_PROCESSOR_NEURAL_NET_CONSUMER_H_
#define STREAMER_PROCESSOR_NEURAL_NET_CONSUMER_H_

#include "common/types.h"
#include "model/model.h"
#include "processor/neural_net_evaluator.h"
#include "processor/processor.h"

// This virtual class serves as an interface for processors that wish to use a
// the output of a NeuralNetEvaluator. A NeuralNetConsumer can either use an
// existing NeuralNetEvaluator or automatically initialize a new
// NeuralNetEvaluator. Any number of sources and sinks are supported. For an
// example of how to implement a NeuralNetConsumer, look at the ImageClassifier
// class.
class NeuralNetConsumer : public Processor {
 public:
  // Automatically constructs a NeuralNetEvaluator, which will be hidden and
  // managed automatically.
  NeuralNetConsumer(ProcessorType type, const ModelDesc& model_desc,
                    const Shape& input_shape,
                    const std::vector<std::string>& output_layer_names = {},
                    const std::vector<std::string>& source_names = {},
                    const std::vector<std::string>& sink_names = {});
  // Assumes that the calling code will construct and connect a
  // NeuralNetEvaluator, which will not be managed automatically.
  NeuralNetConsumer(ProcessorType type,
                    const std::vector<std::string>& source_names = {},
                    const std::vector<std::string>& sink_names = {});

  virtual void SetSource(const std::string& name, StreamPtr stream) override;
  virtual double GetSlidingLatencyMs() const override;
  virtual double GetAvgLatencyMs() const override;
  virtual double GetAvgFps() const override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  // Returns true if this NeuralNetConsumer created the NeuralNetEvaluator
  // that precedes it (meaning that the NeuralNetEvaluator is private and must
  // be managed by the NeuralNetConsumer), or false otherwise.
  bool NneIsPrivate() const;

  std::vector<std::string> output_layer_names_;
  NeuralNetEvaluator* nne_;
};

#endif  // STREAMER_PROCESSOR_NEURAL_NET_CONSUMER_H_
