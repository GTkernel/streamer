//
// Created by Ran Xian (xranthoar@gmail.com) on 9/29/16.
//

#ifndef STREAMER_MODEL_GIE_MODEL_H_
#define STREAMER_MODEL_GIE_MODEL_H_

#include "gie_inferer.h"
#include "model.h"

/**
 * @brief A GIE model
 */
class GIEModel : public Model {
 public:
  GIEModel(const ModelDesc& model_desc, Shape input_shape, int batch_size);
  ~GIEModel();
  virtual void Load();
  virtual void Evaluate();
  virtual void Forward();
  virtual const std::vector<std::string>& GetLayerNames() const override;
  virtual cv::Mat GetLayerOutput(const std::string& layer_name) const override;

 private:
  std::unique_ptr<GIEInferer<float>> inferer_;
};

#endif  // STREAMER_MODEL_GIE_MODEL_H_
