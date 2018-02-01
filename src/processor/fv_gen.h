
#ifndef STREAMER_PROCESSOR_FV_GEN_H_
#define STREAMER_PROCESSOR_FV_GEN_H_

#include <unordered_map>

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"
#include "stream/frame.h"

class FvSpec {
 public:
  FvSpec() {}
  FvSpec(std::string layer_name, int xmin = 0, int ymin = 0, int xmax = 0,
         int ymax = 0, bool flat = true)
      : layer_name_(layer_name),
        roi_(xmin, ymin, xmax - xmin, ymax - ymin),
        yrange_(ymin, ymax),
        xrange_(xmin, xmax),
        xmin_(xmin),
        xmax_(xmax),
        ymin_(ymin),
        ymax_(ymax),
        flat_(flat) {}
  static std::string GetUniqueID(const FvSpec& spec);

 public:
  std::string layer_name_;
  cv::Rect roi_;
  cv::Range yrange_;
  cv::Range xrange_;
  int xmin_, xmax_, ymin_, ymax_;
  bool flat_;
};

// Step 1: construct empty FvGen
// Step 2: call AddFv
class FvGen : public Processor {
 public:
  FvGen();
  ~FvGen();

  void AddFv(std::string layer_name, int xmin = 0, int ymin = 0, int xmax = 0,
             int ymax = 0, bool flat = true);

  static std::shared_ptr<FvGen> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::vector<FvSpec> feature_vector_specs_;
  int honesty_level_;
};

#endif  // STREAMER_PROCESSOR_FV_GEN_H_
