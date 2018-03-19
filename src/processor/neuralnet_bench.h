
#ifndef STREAMER_PROCESSOR_NNBENCH_H_
#define STREAMER_PROCESSOR_NNBENCH_H_

#include <unordered_map>

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"
#include "stream/frame.h"

// A NNBench is a Processor
class NNBench : public Processor {
 public:
  // If output_layer_names is empty, then by default the last layer is
  // published.
  NNBench(const ModelDesc& model_desc, const Shape& input_shape,
          size_t batch_size = 1, int num_classifiers = 1);

  // Returns a vector of the names of this NNBench's sinks, which are
  // the names of the layers that it is publishing.
  const std::vector<std::string> GetSinkNames() const;

  static std::shared_ptr<NNBench> Create(const FactoryParamsType& params);

  // Hides Processor::SetSource(const std::string&, StreamPtr)
  void SetSource(const std::string& name, StreamPtr stream,
                 const std::string& layername = "");
  void SetSource(StreamPtr stream, const std::string& layername = "");
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // Executes the neural network and returns a mapping from the name of a layer
  // to that layer's activations.

  Shape input_shape_;
  std::string input_layer_name_;
  std::vector<std::unique_ptr<Model>> models_;
  std::vector<std::unique_ptr<Frame>> cur_batch_frames_;
  size_t batch_size_;
  int classifiers_;
};

#endif  // STREAMER_PROCESSOR_NNBENCH_H_
