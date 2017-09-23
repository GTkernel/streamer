
#ifndef STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_
#define STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_

#include <mutex>

#include <caffe/caffe.hpp>

#include <Eigen/Dense>

#include "common/types.h"
#include "model/caffe_model.h"
#include "model/model.h"
#include "processor/processor.h"

typedef struct query_t {
  int query_id;
  // indices holds a mask of which images this query uses in the queries_
  // matrix. This could be made more efficient by using an Eigen::VectorXd to
  // hold a binary coefficient matrix used to take a linear combination of rows
  // in queries_.
  std::unique_ptr<Eigen::VectorXf> matches;
  std::unique_ptr<CaffeModel<float>> classifier;
} query_t;

class ImageMatch : public Processor {
 public:
  ImageMatch(unsigned int vishash_size = 1024,
             unsigned int num_hidden_layers = 5, unsigned int batch_size = 1);

  static std::shared_ptr<ImageMatch> Create(const FactoryParamsType& params);

  // Add real query with classifier
  void AddQuery(const std::string& model_path, const std::string& params_path);
  void SetQueryMatrix(int num_queries);
  void SetSink(StreamPtr stream);
  using Processor::SetSink;
  void SetSource(StreamPtr stream);
  using Processor::SetSource;
  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  void AddClassifier(query_t* current_query, const std::string& model_path,
                     const std::string& params_path);
  unsigned int vishash_size_;
  unsigned int num_hidden_layers_;
  unsigned int batch_size_;
  // Hidden layers
  std::shared_ptr<Eigen::MatrixXf> hidden_layer_weights_;
  std::shared_ptr<Eigen::VectorXf> hidden_layer_skews_;
  // Relu
  // Logistic Regression
  std::shared_ptr<Eigen::MatrixXf> logit_weights_;
  std::shared_ptr<Eigen::VectorXf> logit_skews_;
  // Softmax
  // vishash_batch_ stores the vishashes for the current batch of inputs
  std::unique_ptr<Eigen::MatrixXf> vishash_batch_;
  // nn_vishash_batch_ stores the vishashes in cv::Mat form
  std::vector<cv::Mat> nn_vishash_batch_;
  // cur_batch_frames holds the actual frames in the batch
  std::vector<std::unique_ptr<Frame>> frames_batch_;
  std::unordered_map<int, query_t> query_data_;
  std::mutex query_guard_;
};

#endif  // STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_
