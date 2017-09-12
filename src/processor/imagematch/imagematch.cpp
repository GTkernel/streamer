#include "imagematch.h"
#include "model/model_manager.h"
#include "utils/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_def.pb.h>

#include "tensorflow/core/public/session.h"
#include "processor/flow_control/flow_control_entrance.h"

#include <zmq.hpp>

#include <thread>

void ImageMatch::UpdateLinmodMatrix(int query_id) {
    LOG(INFO) << "QUERY " << query_id << " LINMOD READY";
    std::vector<tensorflow::Tensor> linmod_weights;
    TF_CHECK_OK(query_data_[query_id].session_->Run({}, {"var:0", "skew:0"}, {}, &linmod_weights));
    auto weights_map = Eigen::Map<Eigen::VectorXf>(linmod_weights.at(0).flat<float>().data(), 
                                                    linmod_weights.at(0).dim_size(0));
    query_data_[query_id].skew = linmod_weights.at(1).flat<float>()(0);
    query_data_[query_id].linmod_ready = true;
    if(linear_model_weights_ == NULL || query_data_.size() > (decltype(query_data_.size()))linear_model_weights_->rows()) {
      if(linear_model_weights_) {
          delete linear_model_weights_;
      }
      linear_model_weights_ = new Eigen::MatrixXf(query_data_.size(), linmod_weights.at(0).dim_size(0));
    }
    linear_model_weights_->row(query_id) = weights_map;
}

ImageMatch::ImageMatch(std::string linear_model_path, bool do_linmod, unsigned int batch_size)
    : Processor(PROCESSOR_TYPE_CUSTOM, {"input"}, {"output"}),
      batch_size_(batch_size),
      queries_(NULL),
      linear_model_path_(linear_model_path),
      linear_model_weights_(NULL),
      vishash_batch_(NULL),
      do_linmod_(do_linmod),
      linmod_ready_(false) {}

bool ImageMatch::AddQuery(std::string path, std::vector<float> vishash, int query_id, bool is_positive) {
  std::lock_guard<std::mutex> guard(query_guard);
  query_t* current_query = &query_data_[query_id];
  current_query->scores = new Eigen::VectorXf(batch_size_);
  LOG(INFO) << "Received image " << path << " to be added to query (ID = " << query_id << ")";
  Eigen::Map<Eigen::VectorXf> vishash_map(vishash.data(), vishash.size());
  if(queries_ == NULL) {
    queries_ = new Eigen::MatrixXf(1, vishash.size());
    queries_->row(0) = vishash_map;
    queries_->row(0).stableNormalize();
    queries_->row(0) *= is_positive ? 1 : -1;
  } else {
    queries_->conservativeResize(queries_->rows() + 1, queries_->cols());
    queries_->row(queries_->rows() - 1) = vishash_map;
    queries_->row(queries_->rows() - 1).stableNormalize();
    queries_->row(queries_->rows() - 1) *= is_positive ? 1 : -1;
  }
  current_query->indices.push_back(queries_->rows() - 1);
  current_query->paths.push_back(path);
  current_query->linmod_ready = false;
  current_query->query_id = query_id;
  create_session(query_id);
  return true;
}

// Fast random query matrix
bool ImageMatch::SetQueryMatrix(int num_queries, int img_per_query, int vishash_size) {
  std::lock_guard<std::mutex> guard(query_guard);
  queries_ = new Eigen::MatrixXf(num_queries * img_per_query, vishash_size);
  queries_->setRandom();
  for(int i = 0; i < num_queries; ++i) {
    query_t* current_query = &query_data_[i];
    current_query->scores = new Eigen::VectorXf(batch_size_);
    create_session(i);
    current_query->linmod_ready = false;
    current_query->query_id = i;
    for(int j = 0; j < img_per_query; ++j) {
      current_query->indices.push_back(i * img_per_query + j);
      current_query->paths.push_back("test");
    }
  }
  return true;
}

bool ImageMatch::Init() {
  return true;
}

bool ImageMatch::OnStop() {
  return true;
}

void ImageMatch::Process() {
  size_t vishash_size;
  Timer endtoend_timer;
  double endtoend_time;
  endtoend_timer.Start();

  auto frame = GetFrame("input");
  if(query_data_.empty()) {
    std::lock_guard<std::mutex> guard(query_guard);
    auto flow_control_entrance = frame->GetFlowControlEntrance();
    if(flow_control_entrance) {
      flow_control_entrance->ReturnToken();
      frame->SetFlowControlEntrance(nullptr);
      return;
    }
  }
  cv::Mat activations = frame->GetValue<cv::Mat>("activations");
  if(cur_batch_ < batch_size_) {
    // Get the vishash out of the frame
    vishash_size = activations.total();
    Eigen::Map<Eigen::VectorXf> vishash_map((float*)activations.data, vishash_size);
    // Initialize the batch matrix if necessary
    // The reason we do it here is because the vishash size is not known beforehand
    if(vishash_batch_ == NULL) {
      vishash_batch_ = new Eigen::MatrixXf(vishash_size, batch_size_);
    }
    // Add new vishash to batch matrix
    vishash_batch_->col(cur_batch_) = vishash_map;
    vishash_batch_->col(cur_batch_).stableNormalize();
    // release token if necessary
    if(frame->GetFlowControlEntrance()) {
      frame->GetFlowControlEntrance()->ReturnToken();
      frame->SetFlowControlEntrance(nullptr);
    }
    cur_batch_frames_.push_back(std::move(frame));

    // Increment batch counter
    ++cur_batch_;
    // Since the batch isn't full yet, return and wait for next frame
    if(cur_batch_ < batch_size_) {
      return;
    }
  }
  cur_batch_ = 0;

  // Calculate similarity using full formula
  std::lock_guard<std::mutex> guard(query_guard);
  
  Timer matrix_timer;
  double matrix_time;
  matrix_timer.Start();

  Eigen::MatrixXf productmat;
  if(linmod_ready_) {
    productmat = (*linear_model_weights_) * (*vishash_batch_);
  } else {
    productmat = (*queries_) * (*vishash_batch_);
  }

  matrix_time = matrix_timer.ElapsedMicroSec();
  
  Timer add_timer;
  double add_time;
  add_timer.Start();

  for(auto it = query_data_.begin(); it != query_data_.end(); ++it) {
    CHECK(it->second.scores != nullptr);
    it->second.scores->setZero();
    if(linmod_ready_) {
      *(it->second.scores) = (productmat.row(it->second.query_id).array() + it->second.skew).matrix();
    } else {
      for(auto idx = it->second.indices.begin(); idx != it->second.indices.end(); ++idx) {
        *(it->second.scores) += productmat.row(*idx);
      }
    }
    // Normalize
    *(it->second.scores) = (it->second.scores->array() / query_data_.size()).matrix();
  }

  add_time = add_timer.ElapsedMicroSec();

  Timer linmod_timer;
  double linmod_time;
  linmod_timer.Start();
  bool all_ready = true;
  for(auto it = query_data_.begin(); it != query_data_.end(); ++it) {
    if(do_linmod_) {
      tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<long long int>(vishash_size)}));
      std::copy_n((float*)activations.data, vishash_size,
                  x.flat<float>().data());

      tensorflow::Tensor expected(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
      auto expected_flat = expected.flat<float>();
      expected_flat.data()[0] = (*(it->second.scores))(0);

      std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
      inputs.push_back(std::make_pair("x", x));
      inputs.push_back(std::make_pair("expected:0", expected));
      std::vector<tensorflow::Tensor> outputs;

      TF_CHECK_OK(it->second.session_->Run(inputs, {}, {"train:0"}, &outputs));

      TF_CHECK_OK(it->second.session_->Run(inputs, {"actual:0", "loss:0"}, {}, &outputs));
      //LOG(INFO) << "expected: " << inputs.at(1).second.flat<float>()(0) << " actual: " << outputs.at(0).flat<float>() << std::endl;
      //LOG(INFO) << "iteration: " << counter++ << " loss: " << outputs.at(1).flat<float>()(0) << " accuracy: " << (1 - outputs.at(1).flat<float>()(0) / outputs.at(0).flat<float>()(0)) * 100 << " %" << std::endl;
      float loss = outputs.at(1).flat<float>()(0);
      float actual_score = outputs.at(0).flat<float>()(0);
      float expected_score = inputs.at(1).second.flat<float>()(0);
      float difference = expected_score - actual_score;
      if(difference < 0)
        difference *= -1;
      if(loss < 0.001 && !it->second.linmod_ready) {
        UpdateLinmodMatrix(it->second.query_id);
      }
      all_ready &= it->second.linmod_ready;
    }
    linmod_ready_ = all_ready;
  }
  linmod_time = linmod_timer.ElapsedMicroSec();

  std::ostringstream similarity_summary;
  similarity_summary.precision(3);
  if(strlen(similarity_summary.str().c_str()) == 0) {
    similarity_summary << "No queries!";
  }
  int batch_idx = 0;
  for(auto& frame : cur_batch_frames_) {
    std::vector<std::pair<int, float>> image_match_scores;
    for(auto& query : query_data_) {
      std::pair<int, float> score_pair(query.first, (*(query.second.scores))(batch_idx));
      image_match_scores.push_back(score_pair);
    }
    batch_idx += 1;
    frame->SetValue("ImageMatchScores", image_match_scores);
    endtoend_time = endtoend_timer.ElapsedMicroSec();
    frame->SetValue("ImageMatch.Benchmark.EndToEnd", endtoend_time);
    frame->SetValue("ImageMatch.Benchmark.MatrixMultiply", matrix_time);
    frame->SetValue("ImageMatch.Benchmark.GatherAndAdd", add_time);
    frame->SetValue("ImageMatch.Benchmark.LinearModelTrain", linmod_time);
    PushFrame("output", std::move(frame));
  }
  cur_batch_frames_.clear();
  return;
}

// Create linear classifier for query with id = query_id
void ImageMatch::create_session(int query_id) {
  LOG(INFO) << "Creating session";
  query_t* current_query = &query_data_[query_id];
  tensorflow::GraphDef graph_def;
  tensorflow::Status status = ReadBinaryProto(
      tensorflow::Env::Default(), linear_model_path_, &graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to load tf graph at model desc path: "
               << status.error_message().c_str();
  }
  current_query->session_.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  status = current_query->session_->Create(graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to create tf graph session";
  }
  TF_CHECK_OK(current_query->session_->Run({}, {}, {"init"}, nullptr));
}

