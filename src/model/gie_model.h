//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#ifndef STREAMER_GIE_MODEL_H
#define STREAMER_GIE_MODEL_H

#include "gie_inferer.h"
#include "model.h"

/**
 * @brief A GIE model
 */
class GIEModel : public Model {
 public:
  GIEModel(const ModelDesc &model_desc, Shape input_shape, int batch_size);
  ~GIEModel();
  virtual void Load();
  virtual void Evaluate();
  virtual void Forward();

 private:
  std::unique_ptr<GIEInferer<float>> inferer_;
};

#endif  // STREAMER_GIE_MODEL_H
