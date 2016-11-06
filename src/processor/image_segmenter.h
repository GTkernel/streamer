#include "model/model.h"
#include "processor.h"

#ifndef STREAMER_IMAGE_SEGMENTATION_PROCESSOR_H
#define STREAMER_IMAGE_SEGMENTATION_PROCESSOR_H

class ImageSegmenter : public Processor {
 public:
  ImageSegmenter(std::shared_ptr<Stream> input_stream,
                 const ModelDesc &model_desc, Shape input_shape);
  virtual ProcessorType GetType() override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  DataBuffer input_buffer_;
  std::unique_ptr<Model> model_;
  ModelDesc model_desc_;
  Shape input_shape_;
  cv::Mat mean_image_;
};

#endif