
#ifndef STREAMER_MODEL_TF_MODEL_H_
#define STREAMER_MODEL_TF_MODEL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <tensorflow/core/public/session.h>
#include <opencv2/opencv.hpp>

#include "common/types.h"
#include "model/model.h"

class TFModel : public Model {
 public:
  TFModel(const ModelDesc& model_desc, Shape input_shape);
  virtual ~TFModel() override;
  virtual void Load() override;
  virtual std::unordered_map<std::string, cv::Mat> Evaluate(
      const std::unordered_map<std::string, cv::Mat>& input_map,
      const std::vector<std::string>& output_layer_names) override;

 private:
  std::unique_ptr<tensorflow::Session> session_;
  std::vector<std::string> layers_;
  std::string input_op_;
  std::string last_op_;
};

#endif  // STREAMER_MODEL_TF_MODEL_H_
