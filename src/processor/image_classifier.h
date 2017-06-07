
#ifndef STREAMER_IMAGE_CLASSIFIER_H
#define STREAMER_IMAGE_CLASSIFIER_H

#include "common/types.h"
#include "model/model.h"
#include "processor/neural_net_consumer.h"

// An ImageClassifier receives input from a NeuralNetEvaluator (which may be
// hidden) that produces classification label probabilities and matches labels
// to the probabilities. An ImageClassifier has one source named "input" and one
// sink named "output".
class ImageClassifier : public NeuralNetConsumer {
 public:
  // Constructs a NeuralNetEvaluator, which is connected and managed
  // automatically.
  ImageClassifier(const ModelDesc &model_desc, const Shape &input_shape,
                  size_t num_labels = 5);
  // Relies on the calling code to connect this ImageClassifier to an existing
  // NeuralNetEvaluator, which is not managed automatically.
  ImageClassifier(const ModelDesc &model_desc, size_t num_labels = 5);

  virtual ProcessorType GetType() const override;

 protected:
  virtual bool Init() override;
  virtual void Process() override;

 private:
  // Loads the specified model's labels from disk and returns them in a vector.
  static std::vector<std::string> LoadLabels(const ModelDesc &model_desc);

  // The number of labels that will be assigned to each frame.
  size_t num_labels_;
  // A list of all labels, from which num_labels_ entries will be assigned to
  // each frame.
  std::vector<std::string> labels_;
};

#endif  // STREAMER_IMAGE_CLASSIFIER_H
