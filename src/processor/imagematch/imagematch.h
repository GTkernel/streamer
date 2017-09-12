#ifndef STREAMER_IMAGEMATCH_H
#define STREAMER_IMAGEMATCH_H

#include <mutex>

#include "common/common.h"
#include "model/model.h"
#include "processor/processor.h"

#include <Eigen/Dense>

#include "tensorflow/core/public/session.h"

typedef struct query_t {
  int query_id;
  // indices holds a mask of which images this query uses in the queries_ matrix.
  // This could be made more efficient by using an Eigen::VectorXd to hold a
  // binary coefficient matrix used to take a linear combination of rows in
  // queries_.
  std::vector<int> indices;
  std::vector<bool> is_positive;
  std::vector<std::string> paths;
  Eigen::VectorXf* scores;
  float skew;
  std::unique_ptr<tensorflow::Session> session_;
  bool linmod_ready;
} query_t;

class ImageMatch: public Processor {
 public:
  ImageMatch(std::string linear_model_path, bool do_linmod = true, unsigned int batch_size = 1);

  static std::shared_ptr<ImageMatch> Create(
      const FactoryParamsType& params);
  
  void UpdateLinmodMatrix(int query_id);
  bool AddQuery(std::string path, std::vector<float> vishash, int query_id, bool is_positive);
  bool SetQueryMatrix(int num_queries, int img_per_query, int vishash_size);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;
  void create_session(int query_number);

 private:
  unsigned int batch_size_;
  Eigen::MatrixXf* queries_;
  std::string linear_model_path_;
  Eigen::MatrixXf* linear_model_weights_;
  // vishash_batch_ stores the vishashes for the current batch of inputs
  Eigen::MatrixXf* vishash_batch_;
  bool do_linmod_;
  // cur_batch holds the current size of the batch
  unsigned int cur_batch_ = 0;
  // cur_batch_frames holds the actual frames in the batch
  std::vector<std::unique_ptr<Frame>> cur_batch_frames_;
  bool linmod_ready_;

  std::unordered_map<int, query_t> query_data_;
  std::mutex query_guard;

};

#endif  // STREAMER_IMAGEMATCH_H
