
#ifndef STREAMER_PROCESSOR_IMAGE_SEGMENTER_H_
#define STREAMER_PROCESSOR_IMAGE_SEGMENTER_H_

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"

class ImageSegmenter : public Processor {
 public:
  ImageSegmenter(const ModelDesc& model_desc, Shape input_shape);

  static std::shared_ptr<ImageSegmenter> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::unique_ptr<Model> model_;
  ModelDesc model_desc_;
  Shape input_shape_;
  cv::Mat mean_image_;
};

#endif  // STREAMER_PROCESSOR_IMAGE_SEGMENTER_H_
