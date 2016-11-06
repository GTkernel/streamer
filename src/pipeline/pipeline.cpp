//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#include "pipeline.h"

Pipeline::Pipeline() {}

std::shared_ptr<Processor> Pipeline::GetProcessor(const string &name) {
  return std::shared_ptr<Processor>();
}
