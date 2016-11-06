//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#ifndef STREAMER_PIPELINE_H
#define STREAMER_PIPELINE_H

#include "common/common.h"
#include "processor/processor.h"

#include <unordered_map>

class Pipeline {
 public:
  Pipeline();
  std::shared_ptr<Processor> GetProcessor(const string &name);

 private:
  std::unordered_map<string, std::shared_ptr<Processor>> processors_;
};

#endif  // STREAMER_PIPELINE_H
