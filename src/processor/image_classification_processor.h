//
// Created by Ran Xian (xranthoar@gmail.com) on 10/2/16.
//

#ifndef TX1DNN_IMAGE_CLASSIFICATION_PROCESSOR_H
#define TX1DNN_IMAGE_CLASSIFICATION_PROCESSOR_H

#include "processor.h"
#include "common/common.h"
#include "model/model.h"

class ImageClassificationProcessor : public Processor {
 public:
  ImageClassificationProcessor(std::shared_ptr<Stream> input_stream,
                               const ModelDesc &model_desc,
                               Shape input_shape);
 protected:
  virtual bool Init();
  virtual bool OnStop();
  virtual void Consume();

 private:
  /**
   * @brief Transform the image to a given shape, substract mean image, and
   * store to a buffer.
   * @details Specify buffer as nullptr if do not want to store it.
   *
   * @param img The image to be transformed;
   * @param shape The shape of the desired image.
   * @param mean_img Mean image.
   * @param buffer The buffer to store the transformed image.
   * @return The transformed image.
   */
  static cv::Mat TransformImage(const cv::Mat &src_img, const Shape &shape,
                                const cv::Mat &mean_img, DataBuffer *buffer);

  /**
   * @brief Classify the image stored in the input_buffer_
   *
   * @param N The number of predictions desired.
   *
   * @return An array of prediction results, sorted by confidence score.
   */
  std::vector<Prediction> Classify(int N = 5);

  /**
   * @brief Preprocess the image and store to a buffer.
   *
   * @param img The image to be preprocessed.
   * @param buffer The buffer to store the preprocessed result.
   */
  void Preprocess(const cv::Mat &img, DataBuffer &buffer);

  DataBuffer input_buffer_;
  std::unique_ptr<Model> model_;
  std::vector<string> labels_;
  ModelDesc model_desc_;
  Shape input_shape_;
  cv::Mat mean_image_;
};

#endif //TX1DNN_IMAGE_CLASSIFICATION_PROCESSOR_H
