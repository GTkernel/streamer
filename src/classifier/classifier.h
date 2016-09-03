//
// Created by Ran Xian on 8/5/16.
//

#ifndef TX1DNN_CLASSIFIER_H
#define TX1DNN_CLASSIFIER_H

#include "common/common.h"
#include "common/data_buffer.h"

/**
 * @brief The classifier base class.
 * TODO: In the future may consider store the mean image in a unified format so that subclasses do not need to construct the mean image.
 */
class Classifier {
 public:
  /**
   * @brief Initialize the classifier and load the labels.
   * 
   * @param label_file The labels file, one label per line.
   */
  Classifier(const string &model_desc,
             const string &model_params,
             const string &mean_file,
             const string &label_file);

  /**
   * @brief Classify an image, return top N predictions.
   * 
   * @param img The image to be classified.
   * @param N The number of predictions desired.
   * 
   * @return An array of prediction results, sorted by confidence score.
   */
  std::vector <Prediction> Classify(const cv::Mat &img, int N = 5);

  /**
   * @brief Classify an image by given a input buffer directly, this will save preprocessing in the main thread.
   * 
   * @param input_buffer The buffer that stores the preprocessed input.
   * @param N The number of predictions desired.
   * 
   * @return An array of prediction results, sorted by confidence score.
   */
  std::vector <Prediction> Classify(const DataBuffer &input_buffer, int N = 5);

  /**
   * @brief Preprocess the image and store to a buffer.
   * 
   * @param img The image to be preprocessed.
   * @param buffer The buffer to store the preprocessed result.
   */
  virtual void Preprocess(const cv::Mat &img, DataBuffer &buffer);

  /**
   * @brief Transform the image to a given shape, substract mean image, and store to a buffer.
   * @details Specify buffer as nullptr if do not want to store it.
   * 
   * @param img The image to be transformed;
   * @param shape The shape of the desired image.
   * @param mean_img Mean image.
   * @param buffer The buffer to store the transformed image.
   * @return The transformed image.
   */
  static cv::Mat TransformImage(const cv::Mat &img,
                                const Shape &shape,
                                const cv::Mat &mean_img,
                                DataBuffer *buffer);

  /**
   * // TODO: May want to refactor it, expose a public method only for a specific optimization seems terrible.
   * @brief Get the size of the input buffer, used for video pipeline to preprocess image frames.
   * @details It is not supposed to use for any other purporses.
   * @return The size (number of bytes) of the input buffer.
   */
  virtual size_t GetInputBufferSize() = 0;

 private:
  virtual std::vector<float> Predict() = 0;
  virtual DataBuffer GetInputBuffer() = 0;

 protected:
  template<typename T>
  inline size_t GetInputSize() const {
    return input_width_ * input_height_ * input_channels_ * sizeof(T);
  }

  inline Shape GetInputShape() const {
    return Shape(input_channels_, input_width_, input_height_);
  }

  inline cv::Size GetInputGeometry() const {
    return cv::Size(input_width_, input_height_);
  }

  std::vector <string> labels_;
  int input_channels_;
  int input_width_;
  int input_height_;
  cv::Mat mean_image_;
};

#endif //TX1DNN_CLASSIFIER_H
