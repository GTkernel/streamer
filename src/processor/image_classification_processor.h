//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef TX1DNN_IMAGE_CLASSIFICATION_PROCESSOR_H
#define TX1DNN_IMAGE_CLASSIFICATION_PROCESSOR_H

#include "common/common.h"
#include "model/model.h"
#include "processor.h"

class ImageClassificationProcessor : public Processor {
 public:
  ImageClassificationProcessor(std::shared_ptr<Stream> input_stream,
                               std::shared_ptr<Stream> img_stream,
                               const ModelDesc &model_desc, Shape input_shape);

 protected:
  virtual bool Init();
  virtual bool OnStop();
  virtual void Process();

 private:
  /**
   * @brief Classify the image stored in the input_buffer_
   *
   * @param N The number of predictions desired.
   *
   * @return An array of prediction results, sorted by confidence score.
   */
  std::vector<Prediction> Classify(int N = 5);

  DataBuffer input_buffer_;
  std::unique_ptr<Model> model_;
  std::vector<string> labels_;
  ModelDesc model_desc_;
  Shape input_shape_;
  cv::Mat mean_image_;
};

#endif  // TX1DNN_IMAGE_CLASSIFICATION_PROCESSOR_H
