//
// Created by Ran Xian (xranthoar@gmail.com) on 10/21/16.
//

#ifndef STREAMER_DUMMY_NN_PROCESSOR_H
#define STREAMER_DUMMY_NN_PROCESSOR_H

#include "model/model.h"
#include "processor.h"

/**
 * @brief A processor that only runs the forward pass of a given network, do
 * nothing else. It is useful to benchmark the performance of a specific network
 * of interest.
 */
class DummyNNProcessor : public Processor {
 public:
  DummyNNProcessor(const ModelDesc& model_desc);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::unique_ptr<Model> model_;
  ModelDesc model_desc_;
  Shape input_shape_;
  // The fake input of the network
  DataBuffer fake_input_;
};

#endif  // STREAMER_DUMMY_NN_PROCESSOR_H
