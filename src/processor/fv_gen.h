
#ifndef STREAMER_PROCESSOR_FV_GEN_H_
#define STREAMER_PROCESSOR_FV_GEN_H_

#include <unordered_map>

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"
#include "stream/frame.h"

class FVGen : public Processor {
 public:
  FVGen(int xmin, int xmax, int ymin, int ymax);
  ~FVGen();

  static std::shared_ptr<FVGen> Create(const FactoryParamsType& params);

  // Hides Processor::SetSource(const std::string&, StreamPtr)
  void SetSource(const std::string& name, StreamPtr stream);
  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  cv::Rect crop_roi_;
};

#endif  // STREAMER_PROCESSOR_FV_GEN_H_
