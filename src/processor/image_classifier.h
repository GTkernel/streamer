//
// Created by Ran Xian (xranthoar@gmail.com) on 10/6/16.
//

#ifndef STREAMER_BATCH_CLASSIFIER_H
#define STREAMER_BATCH_CLASSIFIER_H

#include "common/common.h"
#include "model/model.h"
#include "processor.h"

class ImageClassifier : public Processor {
 public:
  ImageClassifier(std::vector<std::shared_ptr<Stream>> input_streams,
                  const ModelDesc &model_desc, Shape input_shape);
  virtual ProcessorType GetType() override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  /**
   * @brief Classify the image stored in the input_buffer_
   *
   * @param N The number of predictions desired.
   *
   * @return An array of prediction results, sorted by confidence score.
   */
  std::vector<std::vector<Prediction>> Classify(int N = 5);

  DataBuffer input_buffer_;
  std::unique_ptr<Model> model_;
  std::vector<string> labels_;
  ModelDesc model_desc_;
  Shape input_shape_;
  cv::Mat mean_image_;
  size_t batch_size_;
};

#endif  // STREAMER_BATCH_CLASSIFIER_H
