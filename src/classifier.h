//
// Created by Ran Xian on 8/5/16.
//

#ifndef TX1DNN_CLASSIFIER_H
#define TX1DNN_CLASSIFIER_H

#include "common.h"
#include "data_buffer.h"

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
  Classifier(const string &model_desc, const string &model_params, const string &mean_file, const string& label_file);

  /**
   * @brief Classify an image, return top N predictions.
   * 
   * @param img The image to be classified.
   * @param N The number of predictions desired.
   * 
   * @return An array of prediction results, sorted by confidence score.
   */
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

  /**
   * @brief Classify an image by given a input buffer directly, this will save the step of preprocessing in the critical path.
   * 
   * @param input_buffer The input buffer to store the preprocessing result.
   * @param N The number of predictions desired.
   * 
   * @return An array of prediction results, sorted by confidence score.
   */
  std::vector<Prediction> Classify(const DataBuffer &input_buffer, int N = 5);
  
  /**
   * @brief Preprocess the image, and store to a buffer.
   * 
   * @param img The image to be preprocessed.
   * @param buffer The buffer to store the preprocessed result.
   */
  virtual void Preprocess(const cv::Mat &img, DataBuffer &buffer);

  static cv::Mat TransformImage(const cv::Mat &img, const Shape &shape, const cv::Mat &mean_img, DataBuffer *buffer);

 private:
  virtual std::vector<float> Predict() = 0;
  virtual DataBuffer GetInputBuffer() = 0;

 protected:
  template <typename T>
  inline size_t GetInputSize() const {
    return input_width_ * input_height_ * input_channels_ * sizeof(T);
  }


  inline Shape GetInputShape() const {
    return Shape(input_channels_, input_width_, input_height_);
  }

  inline cv::Size GetInputGeometry() const {
    return cv::Size(input_width_, input_height_);
  }

  std::vector<string> labels_;
  int input_channels_;
  int input_width_;
  int input_height_;
  cv::Mat mean_image_;
};

#endif //TX1DNN_CLASSIFIER_H
