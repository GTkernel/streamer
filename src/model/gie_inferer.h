//
// Created by Ran Xian on 7/26/16.
//

#ifndef STREAMER_MODEL_GIE_INFERER_H_
#define STREAMER_MODEL_GIE_INFERER_H_

#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "common/common.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
/**
 * @brief Class for GIE (GPU Inference Engine)
 */
template <typename DType>
class GIEInferer {
 public:
  typedef std::stringstream GIEModelStreamType;

  class Logger : public ILogger {
   public:
    virtual void log(Severity severity, const char* msg) override;
  };

  static const size_t MAX_WORKSPACE_SIZE = 16 << 20;

 public:
  GIEInferer(const string& deploy_file, const string& model_file,
             const string& input_blob_name_, const string& output_blob_name_,
             int batch_size = 1, bool fp16_mode = false);
  void CreateEngine();
  void DestroyEngine();
  void DoInference(DType* input, DType* output);
  Shape GetInputShape();
  Shape GetOutputShape();

 private:
  void CaffeToGIEModel(const string& deploy_file, const string& model_file,
                       const std::vector<string>& outputs,
                       unsigned int max_batch_size,
                       std::ostream& gie_model_stream);

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
  IRuntime* infer_runtime_;
  ICudaEngine* engine_;
  GIEModelStreamType gie_model_stream_;
  DType* d_input_buffer_;
  DType* d_output_buffer_;

  int batch_size_;
  bool fp16_mode_;
};

#endif  // STREAMER_MODEL_GIE_INFERER_H_
