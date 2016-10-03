//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#ifndef TX1DNN_GIE_MODEL_H
#define TX1DNN_GIE_MODEL_H

#include "model.h"
#include "gie_inferer.h"

/**
 * @brief A GIE model
 */
class GIEModel : public Model {
  typedef float DType;
 public:
  GIEModel(const ModelDesc &model_desc, Shape input_shape);
  ~GIEModel();
  virtual void Load();
  virtual void Evaluate();
 private:
  std::unique_ptr<GIEInferer<DType>> inferer_;
};

#endif //TX1DNN_GIE_MODEL_H
