
#include "processor/imagematch/imagematch.h"

#include <string>
#include <thread>

#ifdef USE_TENSORFLOW
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_def.pb.h>
#include <tensorflow/core/public/session.h>
#endif  // USE_TENSORFLOW

#include <zmq.hpp>

#include "common/common.h"
#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "utils/utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

ImageMatch::ImageMatch(const std::string& linear_model_path, bool do_linmod,
                       unsigned int batch_size)
    : Processor(PROCESSOR_TYPE_IMAGEMATCH, {SOURCE_NAME}, {SINK_NAME}),
#ifdef USE_TENSORFLOW
      linear_model_path_(linear_model_path),
      linear_model_weights_(nullptr),
      do_linmod_(do_linmod),
      linmod_ready_(false),
#endif  // USE_TENSORFLOW
      batch_size_(batch_size),
      queries_(nullptr),
      vishash_batch_(nullptr) {
  (void)linear_model_path;
  (void)do_linmod;
}

std::shared_ptr<ImageMatch> ImageMatch::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

#ifdef USE_TENSORFLOW
void ImageMatch::UpdateLinmodMatrix(int query_id) {
  std::vector<tensorflow::Tensor> linmod_weights;
  TF_CHECK_OK(query_data_[query_id].session_->Run({}, {"var:0", "skew:0"}, {},
                                                  &linmod_weights));
  auto weights_map =
      Eigen::Map<Eigen::VectorXf>(linmod_weights.at(0).flat<float>().data(),
                                  linmod_weights.at(0).dim_size(0));
  query_data_[query_id].skew = linmod_weights.at(1).flat<float>()(0);
  query_data_[query_id].linmod_ready = true;
  if (linear_model_weights_ == NULL ||
      query_data_.size() >
          (decltype(query_data_.size()))linear_model_weights_->rows()) {
    linear_model_weights_ = std::make_unique<Eigen::MatrixXf>(
        query_data_.size(), linmod_weights.at(0).dim_size(0));
  }
  linear_model_weights_->row(query_id) = weights_map;
}
#endif  // USE_TENSORFLOW

bool ImageMatch::AddQuery(const std::string& path, std::vector<float> vishash,
                          int query_id, bool is_positive, float threshold) {
  std::lock_guard<std::mutex> guard(query_guard_);
  query_t* current_query = &query_data_[query_id];
  current_query->scores = std::make_unique<Eigen::VectorXf>(batch_size_);
  LOG(INFO) << "Received image " << path
            << " to be added to query (ID = " << query_id << ")";
  Eigen::Map<Eigen::VectorXf> vishash_map(vishash.data(), vishash.size());
  if (queries_ == NULL) {
    queries_ = std::make_unique<Eigen::MatrixXf>(1, vishash.size());
    queries_->row(0) = vishash_map;
    queries_->row(0).stableNormalize();
    queries_->row(0) *= is_positive ? 1 : -1;
  } else {
    queries_->conservativeResize(queries_->rows() + 1, Eigen::NoChange);
    queries_->row(queries_->rows() - 1) = vishash_map;
    queries_->row(queries_->rows() - 1).stableNormalize();
    queries_->row(queries_->rows() - 1) *= is_positive ? 1 : -1;
  }
  current_query->indices.push_back(queries_->rows() - 1);
  current_query->paths.push_back(path);
  current_query->query_id = query_id;
  current_query->threshold = threshold;
#ifdef USE_TENSORFLOW
  current_query->linmod_ready = false;
  CreateSession(query_id);
#endif  // USE_TENSORFLOW
  return true;
}

void ImageMatch::SetQueryMatrix(std::shared_ptr<Eigen::MatrixXf> matrix, float threshold) {
  queries_ = matrix;
  for (int i = 0; i < matrix->rows(); ++i) {
    query_t* current_query = &query_data_[i];
    current_query->scores = std::make_unique<Eigen::VectorXf>(batch_size_);
#ifdef USE_TENSORFLOW
    CreateSession(i);
    current_query->linmod_ready = false;
#endif  // USE_TENSORFLOW
    current_query->query_id = i;
    current_query->threshold = threshold;
    current_query->indices.push_back(i);
    current_query->paths.push_back("test");
  }
}

// Fast random query matrix
bool ImageMatch::SetQueryMatrix(int num_queries, int img_per_query,
                                int vishash_size, float threshold) {
  std::lock_guard<std::mutex> guard(query_guard_);
  if(queries_ == nullptr) {
    queries_ = std::make_unique<Eigen::MatrixXf>(num_queries * img_per_query,
                                                 vishash_size);
    queries_->setRandom();
  } else {
    int old_num_rows = queries_->rows();
    CHECK(num_queries > old_num_rows) << "Removing queries is not yet supported";
    queries_->conservativeResize(num_queries * img_per_query, Eigen::NoChange);
    queries_->block(old_num_rows, queries_->rows(), queries_->rows() - old_num_rows, queries_->cols()).setRandom();
  }
  for (int i = query_data_.size(); i < num_queries; ++i) {
    query_t* current_query = &query_data_[i];
    current_query->scores = std::make_unique<Eigen::VectorXf>(batch_size_);
#ifdef USE_TENSORFLOW
    CreateSession(i);
    current_query->linmod_ready = false;
#endif  // USE_TENSORFLOW
    current_query->query_id = i;
    current_query->threshold = threshold;
    for (int j = 0; j < img_per_query; ++j) {
      current_query->indices.push_back(i * img_per_query + j);
      current_query->paths.push_back("test");
    }
  }
  return true;
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
  size_t vishash_size;
  auto start_time = boost::posix_time::microsec_clock::local_time();

  auto frame = GetFrame("input");
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
  if (cur_batch_ < batch_size_) {
    // Get the vishash out of the frame
    vishash_size = activations.total();
    Eigen::Map<Eigen::VectorXf> vishash_map((float*)activations.data,
                                            vishash_size);
    // Initialize the batch matrix if necessary
    // The reason we do it here is because the vishash size is not known
    // beforehand
    if (vishash_batch_ == NULL) {
      vishash_batch_ =
          std::make_unique<Eigen::MatrixXf>(vishash_size, batch_size_);
    }
    // Add new vishash to batch matrix
    vishash_batch_->col(cur_batch_) = vishash_map;
    vishash_batch_->col(cur_batch_).stableNormalize();
    cur_batch_frames_.push_back(std::move(frame));

    // Increment batch counter
    ++cur_batch_;
    // Since the batch isn't full yet, return and wait for next frame
    if (cur_batch_ < batch_size_) {
      return;
    }
  }
  cur_batch_ = 0;

  // Calculate similarity using full formula
  std::lock_guard<std::mutex> guard(query_guard_);
  auto overhead_end_time = boost::posix_time::microsec_clock::local_time();

  Eigen::MatrixXf productmat;
#ifdef USE_TENSORFLOW
  if (linmod_ready_) {
    productmat = (*linear_model_weights_) * (*vishash_batch_);
  } else {
#endif  // USE_TENSORFLOW
    productmat = (*queries_) * (*vishash_batch_);
#ifdef USE_TENSORFLOW
  }
#endif  // USE_TENSORFLOW

  auto matrix_end_time = boost::posix_time::microsec_clock::local_time();

  for (auto it = query_data_.begin(); it != query_data_.end(); ++it) {
    it->second.scores->setZero();
#ifdef USE_TENSORFLOW
    if (linmod_ready_) {
      *(it->second.scores) =
          (productmat.row(it->second.query_id).array() + it->second.skew)
              .matrix();
    } else {
#endif  // USE_TENSORFLOW
      for (auto idx = it->second.indices.begin();
           idx != it->second.indices.end(); ++idx) {
        *(it->second.scores) += productmat.row(*idx);
      }
#ifdef USE_TENSORFLOW
    }
#endif  // USE_TENSORFLOW
    // Normalize
    *(it->second.scores) =
        (it->second.scores->array() / query_data_.size()).matrix();
  }

  auto add_end_time = boost::posix_time::microsec_clock::local_time();

#ifdef USE_TENSORFLOW
  bool all_ready = true;
  if (do_linmod_) {
    for (auto it = query_data_.begin(); it != query_data_.end(); ++it) {
      tensorflow::Tensor x(
          tensorflow::DT_FLOAT,
          tensorflow::TensorShape({static_cast<long long int>(vishash_size)}));
      std::copy_n((float*)activations.data, vishash_size,
                  x.flat<float>().data());

      tensorflow::Tensor expected(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({1}));
      auto expected_flat = expected.flat<float>();
      expected_flat.data()[0] = (*(it->second.scores))(0);

      std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
      inputs.push_back(std::make_pair("x", x));
      inputs.push_back(std::make_pair("expected:0", expected));
      std::vector<tensorflow::Tensor> outputs;

      TF_CHECK_OK(it->second.session_->Run(inputs, {}, {"train:0"}, &outputs));

      TF_CHECK_OK(it->second.session_->Run(inputs, {"actual:0", "loss:0"}, {},
                                           &outputs));
      float loss = outputs.at(1).flat<float>()(0);
      float actual_score = outputs.at(0).flat<float>()(0);
      float expected_score = inputs.at(1).second.flat<float>()(0);
      float difference = expected_score - actual_score;
      if (difference < 0) difference *= -1;
      if (loss < 0.001 && !it->second.linmod_ready) {
        UpdateLinmodMatrix(it->second.query_id);
      }
      all_ready &= it->second.linmod_ready;
    }
    linmod_ready_ = all_ready;
  }
#endif  // USE_TENSORFLOW
  auto linmod_end_time = boost::posix_time::microsec_clock::local_time();

  std::ostringstream similarity_summary;
  similarity_summary.precision(3);
  if (strlen(similarity_summary.str().c_str()) == 0) {
    similarity_summary << "No queries!";
  }
  int batch_idx = 0;
  for (auto& frame : cur_batch_frames_) {
    std::vector<std::pair<int, float>> image_match_scores;
    for (auto& query : query_data_) {
      if ((*(query.second.scores))(batch_idx) > query.second.threshold) {
        std::pair<int, float> score_pair(query.first,
                                         (*(query.second.scores))(batch_idx));
        image_match_scores.push_back(score_pair);
      }
    }
    batch_idx += 1;
    frame->SetValue("imagematch.scores", image_match_scores);
    auto end_time = boost::posix_time::microsec_clock::local_time();
    frame->SetValue("imagematch.end_to_end_time_micros",
                    (end_time - start_time).total_microseconds());
    frame->SetValue("imagematch.matrix_multiply_time_micros",
                    (matrix_end_time - overhead_end_time).total_microseconds());
    frame->SetValue("imagematch.add_time_micros",
                    (add_end_time - matrix_end_time).total_microseconds());
    frame->SetValue("imagematch.lin_mod_training_time_micros",
                    (linmod_end_time - add_end_time).total_microseconds());
    PushFrame("output", std::move(frame));
  }
  cur_batch_frames_.clear();
  return;
}
#ifdef USE_TENSORFLOW
// Create linear classifier for query with id = query_id
void ImageMatch::CreateSession(int query_id) {
  if (do_linmod_) {
    query_t* current_query = &query_data_[query_id];
    tensorflow::GraphDef graph_def;
    tensorflow::Status status = ReadBinaryProto(tensorflow::Env::Default(),
                                                linear_model_path_, &graph_def);
    if (!status.ok()) {
      LOG(FATAL) << "Failed to load tf graph at model desc path: "
                 << status.error_message().c_str();
    }
    current_query->session_.reset(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    status = current_query->session_->Create(graph_def);
    if (!status.ok()) {
      LOG(FATAL) << "Failed to create tf graph session";
    }
    TF_CHECK_OK(current_query->session_->Run({}, {}, {"init"}, nullptr));
  }
}
#endif  // USE_TENSORFLOW
