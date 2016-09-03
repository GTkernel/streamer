//
// Created by Ran Xian on 7/26/16.
//

#ifndef TX1DNN_GIEINFERER_H
#define TX1DNN_GIEINFERER_H

#include "common/common.h"
#include <Infer.h>
#include <caffeParser.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;
/**
 * @brief Class for GIE (GPU Inference Engine)
 */
template<typename DType>
class GIEInferer {
 public:
  typedef std::stringstream GIEModelStreamType;

  class Logger : public ILogger {
   public:
    virtual void log(Severity severity, const char *msg) override;
  };

  static const size_t MAX_WORKSPACE_SIZE = 16 << 20;
  static const size_t BATCH_SIZE = 1;

 public:
  GIEInferer(const string &deploy_file,
             const string &model_file,
             const string &input_blob_name_,
             const string &output_blob_name_);
  void CreateEngine();
  void DestroyEngine();
  void DoInference(DType *input, DType *output);
  Shape GetInputShape();
  Shape GetOutputShape();

 private:
  void CaffeToGIEModel(const string &deploy_file,
                       const string &model_file,
                       const std::vector<string> &outputs,
                       unsigned int max_batch_size, std::ostream &gie_model_stream);

 private:
  string deploy_file_;
  string model_file_;
  string input_blob_name_;
  string output_blob_name_;
  Shape input_shape_;
  Shape output_shape_;
  size_t input_size_;
  size_t output_size_;
  Logger logger_;
  IRuntime *infer_runtime_;
  ICudaEngine *engine_;
  GIEModelStreamType gie_model_stream_;
  DType *d_input_buffer;
  DType *d_output_buffer;
};


#endif //TX1DNN_GIEINFERER_H
