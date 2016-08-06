//
// Created by Ran Xian on 7/27/16.
//

#ifndef TX1DNN_GIECLASSIFIER_H
#define TX1DNN_GIECLASSIFIER_H

#include "common.h"
#include "classifier.h"
#include "gie_inferer.h"
#include "float16.h"

class GIEClassifier : public Classifier {
  typedef float DType;
 public:
  typedef std::pair<string, float> Prediction;
  GIEClassifier(const string &model_file,
                const string &trained_file,
                const string &mean_file,
                const string &label_file);
  ~GIEClassifier();

 private:
  void SetMean(const string &mean_file);
  virtual std::vector<float> Predict();
  virtual DataBuffer GetInputBuffer();

 private:
  GIEInferer<DType> inferer_;
  DataBuffer input_buffer_;
  DataBuffer output_buffer_;
};

#endif //TX1DNN_GIECLASSIFIER_H
