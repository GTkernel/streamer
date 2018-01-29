
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
    FvSpec(std::string layer_name, int xmin, int xmax, int ymin, int ymax, bool flat) :
        layer_name_(layer_name),
        roi_(xmin, ymin, xmax - xmin, ymax - ymin),
        xmin_(xmin),
        xmax_(xmax),
        ymin_(ymin),
        ymax_(ymax),
        flat_(flat) {}
    static std::string GetUniqueID(const FvSpec& spec);
  public:
    std::string layer_name_;
    cv::Rect roi_;
    int xmin_, xmax_, ymin_, ymax_;
    bool flat_;
};

class FvGen: public Processor {
 public:
  FvGen(int xmin, int ymin, int xmax, int ymax, bool flat = false);
  ~FvGen();

  
  void AddFV(std::string layer_name, int xmin, int ymin, int xmax, int ymax, bool flat = true);

  static std::shared_ptr<FvGen> Create(
      const FactoryParamsType& params);

  // Hides Processor::SetSource(const std::string&, StreamPtr)
  void SetSource(const std::string& name, StreamPtr stream) override;
  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::vector<FvSpec> feature_vector_specs_;
};

#endif  // STREAMER_PROCESSOR_FV_GEN_H_
