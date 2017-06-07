//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#ifndef STREAMER_MXNET_MODEL_H
#define STREAMER_MXNET_MODEL_H

#include <stdlib.h>

#include <mxnet/c_api.h>
#include <mxnet/c_predict_api.h>
#include "common/common.h"
#include "model.h"

/**
 * @brief MXNet model
 */
class MXNetModel : public Model {
 public:
  MXNetModel(const ModelDesc &model_desc, Shape input_shape, int batch_size);
  ~MXNetModel();
  virtual void Load() override;
  virtual void Evaluate() override;
  virtual void Forward();
  virtual const std::vector<std::string> &GetLayerNames() const override;
  virtual cv::Mat GetLayerOutput(const std::string &layer_name) const override;

 private:
  PredictorHandle predictor_;
};

#endif  // STREAMER_MXNET_MODEL_H
