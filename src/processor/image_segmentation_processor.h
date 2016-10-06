#include "model/model.h"
#include "processor.h"

#ifndef TX1DNN_IMAGE_SEGMENTATION_PROCESSOR_H
#define TX1DNN_IMAGE_SEGMENTATION_PROCESSOR_H

class ImageSegmentationProcessor : public Processor {
 public:
  ImageSegmentationProcessor(std::shared_ptr<Stream> input_stream,
                             std::shared_ptr<Stream> img_stream,
                             const ModelDesc &model_desc, Shape input_shape);

 protected:
  virtual bool Init();
  virtual bool OnStop();
  virtual void Process();

 private:
  DataBuffer input_buffer_;
  std::unique_ptr<Model> model_;
  ModelDesc model_desc_;
  Shape input_shape_;
  cv::Mat mean_image_;
};

#endif