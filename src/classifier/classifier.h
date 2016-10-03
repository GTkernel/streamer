//
// Created by Ran Xian on 8/5/16.
//

#ifndef TX1DNN_CLASSIFIER_H
#define TX1DNN_CLASSIFIER_H

#include "common/common.h"
#include "common/data_buffer.h"
#include "model/model.h"
#include "stream/stream.h"

/**
 * @brief The classifier base class.
 * TODO: In the future may consider store the mean image in a unified format so
 * that subclasses do not need to construct the mean image.
 */
class Classifier {
 public:
  /**
   * @brief Initialize the classifier and load the labels.
   *
   * @param model_desc The decription of the model.
   */
  Classifier(std::shared_ptr<Stream> input_stream,
             const ModelDesc &model_desc,
             Shape input_shape);

  void Start();
  void Stop();
 protected:
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
   * @brief Classify an image by given a input buffer directly, this will save
   * preprocessing in the main thread.
   *
   * @param input_buffer The buffer that stores the preprocessed input.
   * @param N The number of predictions desired.
   *
   * @return An array of prediction results, sorted by confidence score.
   */
  std::vector<Prediction> Classify(const DataBuffer &input_buffer, int N = 5);

  /**
   * @brief Preprocess the image and store to a buffer.
   *
   * @param img The image to be preprocessed.
   * @param buffer The buffer to store the preprocessed result.
   */
  virtual void Preprocess(const cv::Mat &img, DataBuffer &buffer);

  std::unique_ptr<Model> model_;
  std::vector<string> labels_;
  Shape input_shape_;
  cv::Mat mean_image_;

  std::shared_ptr<Stream> input_stream_;
  bool stopped_;
};

#endif  // TX1DNN_CLASSIFIER_H
