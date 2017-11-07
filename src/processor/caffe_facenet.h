/**
 * Face feature extractor using facenet
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_CAFFE_FACENET_H_
#define STREAMER_PROCESSOR_CAFFE_FACENET_H_

#include <caffe/caffe.hpp>

#include "model/model.h"
#include "processor/processor.h"

class Facenet : public Processor {
 public:
  Facenet(const ModelDesc& model_desc, Shape input_shape, size_t batch_size);
  static std::shared_ptr<Facenet> Create(const FactoryParamsType& params);
  void SetInputStream(int src_id, StreamPtr stream);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::unique_ptr<caffe::Net<float>> net_;
  void* input_buffer_;
  std::unique_ptr<Model> model_;
  ModelDesc model_desc_;
  Shape input_shape_;
  cv::Mat mean_image_;
  size_t batch_size_;
  size_t face_batch_size_;
  cv::Mat face_image_;
  cv::Mat face_image_resized_;
  cv::Mat face_image_float_;
  cv::Mat face_image_subtract_;
  cv::Mat face_image_normalized_;
  cv::Mat face_image_bgr_;
};

#endif  // STREAMER_PROCESSOR_CAFFE_FACENET_H_
