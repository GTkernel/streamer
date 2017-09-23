
#include "processor/imagematch/imagematch.h"

#include <string>
#include <thread>

#include <zmq.hpp>

#include "common/common.h"
#include "model/caffe_model.h"
#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "utils/utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

ImageMatch::ImageMatch(unsigned int vishash_size, unsigned int num_hidden_layers, unsigned int batch_size)
    : Processor(PROCESSOR_TYPE_IMAGEMATCH, {SOURCE_NAME}, {SINK_NAME}),
     vishash_size_(vishash_size),
     num_hidden_layers_(num_hidden_layers),
     batch_size_(batch_size),
     hidden_layer_weights_(nullptr),
     hidden_layer_skews_(nullptr),
     logit_weights_(nullptr),
     logit_skews_(nullptr) {
  vishash_batch_ = std::make_unique<Eigen::MatrixXf>(batch_size_, vishash_size_);
}

std::shared_ptr<ImageMatch> ImageMatch::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void ImageMatch::AddQuery(const std::string& model_path, const std::string& params_path) {
  std::lock_guard<std::mutex> guard(query_guard_);
  CHECK(hidden_layer_weights_ != nullptr) << "Cannot mix normal and fake queries";
  int query_id = query_data_.size();
  query_t* current_query = &query_data_[query_id];
  current_query->matches = std::make_unique<Eigen::VectorXf>(batch_size_);
  current_query->query_id = query_id;
  AddClassifier(current_query, model_path, params_path);
}

// Fast random query matrix
void ImageMatch::SetQueryMatrix(int num_queries) {
  std::lock_guard<std::mutex> guard(query_guard_);
  if(hidden_layer_weights_ == nullptr) {
    hidden_layer_weights_ = std::make_unique<Eigen::MatrixXf>(vishash_size_, num_hidden_layers_ * num_queries);
    hidden_layer_weights_->setRandom();
  } else {
    int old_num_cols = hidden_layer_weights_->cols();
    hidden_layer_weights_->conservativeResize(Eigen::NoChange, num_hidden_layers_ * num_queries);
    hidden_layer_weights_->block(old_num_cols, hidden_layer_weights_->cols(), hidden_layer_weights_->cols() - old_num_cols, hidden_layer_weights_->cols()).setRandom();
  }
  if(hidden_layer_skews_ == nullptr) {
    hidden_layer_skews_ = std::make_unique<Eigen::VectorXf>(num_hidden_layers_ * num_queries);
    hidden_layer_skews_->setRandom();
  } else {
    hidden_layer_skews_->conservativeResize(num_hidden_layers_ * num_queries);
    hidden_layer_skews_->setRandom();
  }
  if(logit_weights_ == nullptr) {
    logit_weights_ = std::make_unique<Eigen::MatrixXf>(num_hidden_layers_ * num_queries, 2);
    logit_weights_->setRandom();
  } else {
    int old_num_rows = logit_weights_->rows();
    logit_weights_->conservativeResize(num_hidden_layers_ * num_queries, Eigen::NoChange);
    logit_weights_->block(old_num_rows, logit_weights_->rows(), logit_weights_->rows() - old_num_rows, logit_weights_->rows()).setRandom();
  }
  if(logit_skews_ == nullptr) {
    logit_skews_ = std::make_unique<Eigen::VectorXf>(2 * num_queries);
    logit_skews_->setRandom();
  } else {
    logit_skews_->conservativeResize(2 * num_queries);
    logit_skews_->setRandom();
  }
  for (int i = query_data_.size(); i < num_queries; ++i) {
    query_t* current_query = &query_data_[i];
    current_query->matches = std::make_unique<Eigen::VectorXf>(batch_size_);
    current_query->query_id = i;
  }
}

void ImageMatch::SetSink(StreamPtr stream) {
  Processor::SetSink(SINK_NAME, stream);
}

void ImageMatch::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr ImageMatch::GetSink() { return Processor::GetSink(SINK_NAME); }

bool ImageMatch::Init() { return true; }

bool ImageMatch::OnStop() { return true; }

void ImageMatch::Process() {
  auto start_time = boost::posix_time::microsec_clock::local_time();

  auto frame = GetFrame("input");
  CHECK(frame != nullptr);
  // If no queries, release frame
  if (query_data_.empty()) {
    std::lock_guard<std::mutex> guard(query_guard_);
    auto flow_control_entrance = frame->GetFlowControlEntrance();
    if (flow_control_entrance) {
      flow_control_entrance->ReturnToken();
      frame->SetFlowControlEntrance(nullptr);
      return;
    }
  }
  cv::Mat activations = frame->GetValue<cv::Mat>("activations");
  Eigen::Map<Eigen::VectorXf> vishash_map((float*)activations.data,
                                          vishash_size_);
  vishash_batch_->row(frames_batch_.size()) = vishash_map;
  frames_batch_.push_back(std::move(frame));
  if (frames_batch_.size() < batch_size_) {
      return;
  }

  // Calculate similarity using full formula
  std::lock_guard<std::mutex> guard(query_guard_);
  auto overhead_end_time = boost::posix_time::microsec_clock::local_time();

  if(hidden_layer_weights_ != nullptr) {
    // Use fake evaluation
    Eigen::MatrixXf out = ((*vishash_batch_) * (*hidden_layer_weights_)).matrix();
    LOG(INFO) << "input: (" << vishash_batch_->rows() << " x " << vishash_batch_->cols() << ")";
    LOG(INFO) << "hidden: (" << hidden_layer_weights_->rows() << " x " << hidden_layer_weights_->cols() << ")";
    // TODO: issue with broadcasting
    // out.colwise() += (*hidden_layer_skews_);
    LOG(INFO) << "hidden out: (" << out.rows() << " x " << out.cols() << ")";
    // relu max
    out = out.cwiseMax(0);
    LOG(INFO) << "relu out: (" << out.rows() << " x " << out.cols() << ")";
    // logit multiplication and skew
    out *= (*logit_weights_);
    LOG(INFO) << "logit: (" << logit_weights_->rows() << " x " << logit_weights_->cols() << ")";
    LOG(INFO) << "logit weight out: (" << out.rows() << " x " << out.cols() << ")";
    // out.colwise() += (*logit_skews_);
    LOG(INFO) << "logit out: (" << out.rows() << " x " << out.cols() << ")";
    // TODO softmax
  } else {
    // Use real evaluation
    LOG(FATAL) << "Only fake evaluation supported at this time";
  }

  auto matrix_end_time = boost::posix_time::microsec_clock::local_time();
  for (decltype(frames_batch_.size()) batch_idx = 0; batch_idx < frames_batch_.size(); ++batch_idx) {
    std::vector<int> image_match_matches;
    for (auto& query : query_data_) {
      if((*(query.second.matches))(batch_idx) == 1) {
        image_match_matches.push_back(query.second.query_id);
      }
    }
    frames_batch_.at(batch_idx)->SetValue("imagematch.matches", image_match_matches);
    auto end_time = boost::posix_time::microsec_clock::local_time();
    frames_batch_.at(batch_idx)->SetValue("imagematch.end_to_end_time_micros",
                    (end_time - start_time).total_microseconds());
    frames_batch_.at(batch_idx)->SetValue("imagematch.matrix_multiply_time_micros",
                    (matrix_end_time - overhead_end_time).total_microseconds());
    PushFrame("output", std::move(frames_batch_.at(batch_idx)));
  }
  frames_batch_.clear();
  return;
}

void ImageMatch::AddClassifier(query_t* current_query, const std::string& model_path, const std::string& params_path) {
    Shape input_shape(1, 1024);
    ModelDesc model_desc("classifier", MODEL_TYPE_CAFFE, model_path, params_path, 1, 1024, "input", "output");
    current_query->classifier = std::make_unique<CaffeModel<float>>(model_desc, input_shape, batch_size_);
}
