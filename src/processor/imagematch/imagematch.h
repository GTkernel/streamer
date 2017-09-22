
#ifndef STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_
#define STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_

#include <mutex>

#ifdef USE_TENSORFLOW
#include <tensorflow/core/public/session.h>
#endif  // USE_TENSORFLOW
#include <Eigen/Dense>

#include "common/types.h"
#include "model/model.h"
#include "processor/processor.h"

typedef struct query_t {
  int query_id;
  // indices holds a mask of which images this query uses in the queries_
  // matrix. This could be made more efficient by using an Eigen::VectorXd to
  // hold a binary coefficient matrix used to take a linear combination of rows
  // in queries_.
  std::vector<int> indices;
  std::vector<bool> is_positive;
  std::vector<std::string> paths;
  std::unique_ptr<Eigen::VectorXf> scores;
  float threshold;
#ifdef USE_TENSORFLOW
  std::unique_ptr<tensorflow::Session> session_;
  bool linmod_ready;
  float skew;
#endif  // USE_TENSORFLOW
} query_t;

class ImageMatch : public Processor {
 public:
  ImageMatch(const std::string& linear_model_path, bool do_linmod = true,
             unsigned int batch_size = 1);

  static std::shared_ptr<ImageMatch> Create(const FactoryParamsType& params);

#ifdef USE_TENSORFLOW
  void UpdateLinmodMatrix(int query_id);
#endif  // USE_TENSORFLOW
  bool AddQuery(const std::string& path, std::vector<float> vishash,
                int query_id, bool is_positive, float threshold = 1.0);
  bool SetQueryMatrix(int num_queries, int img_per_query, int vishash_size,
                      float threshold = 0.0);
  void SetQueryMatrix(std::shared_ptr<Eigen::MatrixXf> matrix, float threshold = 0.0);
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
  // Linear classifier related members
#ifdef USE_TENSORFLOW
  void CreateSession(int query_number);
  std::string linear_model_path_;
  std::unique_ptr<Eigen::MatrixXf> linear_model_weights_;
  bool do_linmod_;
  bool linmod_ready_;
#endif  // USE_TENSORFLOW
  unsigned int batch_size_;
  std::shared_ptr<Eigen::MatrixXf> queries_;
  // vishash_batch_ stores the vishashes for the current batch of inputs
  std::unique_ptr<Eigen::MatrixXf> vishash_batch_;
  // cur_batch holds the current size of the batch
  unsigned int cur_batch_ = 0;
  // cur_batch_frames holds the actual frames in the batch
  std::vector<std::unique_ptr<Frame>> cur_batch_frames_;
  std::unordered_map<int, query_t> query_data_;
  std::mutex query_guard_;
};

#endif  // STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_
