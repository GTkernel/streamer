
#ifndef STREAMER_APPS_TRAIN_DETECTOR_LITESQL_WRITER_H_
#define STREAMER_APPS_TRAIN_DETECTOR_LITESQL_WRITER_H_

#include <memory>
#include <string>

#include "litesql.hpp"

#include "common/types.h"
#include "processor/processor.h"

class LiteSqlWriter : public Processor {
 public:
  LiteSqlWriter(const std::string& output_dir);

  static std::shared_ptr<LiteSqlWriter> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::string output_dir_;
};

#endif  // STREAMER_APPS_TRAIN_DETECTOR_LITESQL_WRITER_H_
