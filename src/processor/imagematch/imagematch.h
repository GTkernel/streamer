
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
  std::unique_ptr<caffe::Net<float>> classifier;
} query_t;

class ImageMatch : public Processor {
 public:
  ImageMatch(unsigned int vishash_size = 1024, unsigned int batch_size = 1);

  static std::shared_ptr<ImageMatch> Create(const FactoryParamsType& params);

  // Add real query with Micro Classifier
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
  // SetClassifier is a helper function to add/replace a Micro Classifier for
  // an existing query
  void SetClassifier(query_t* current_query, const std::string& model_path,
                     const std::string& params_path);
  // vishash_size_ should hold the number of elements (floats) in the vishash
  unsigned int vishash_size_;
  // Batch size should be above 0
  unsigned int batch_size_;
  // cur_batch_frames holds the actual frames in the batch
  std::vector<std::unique_ptr<Frame>> frames_batch_;
  // query_data_ stores information related to the query
  // as well as state data regarding the most recent run of the
  // Micro Classifier related to this query
  std::unordered_map<int, query_t> query_data_;
  // Lock to prevent race conditions when adding/modifying queries
  std::mutex query_guard_;
};

#endif  // STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_
