// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_
#define STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_

#include "common/types.h"
#include "processor/processor.h"
#include "stream/stream.h"

class ImageTransformer : public Processor {
 public:
  ImageTransformer(const Shape& target_shape, bool crop,
                   unsigned int angle = 0);

  static std::shared_ptr<ImageTransformer> Create(
      const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  Shape target_shape_;
  bool crop_;
  unsigned int angle_;
};

#endif  // STREAMER_PROCESSOR_IMAGE_TRANSFORMER_H_
