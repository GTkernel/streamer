
#include "processor/imagematch/imagematch.h"

#include <string>
#include <thread>

#include <zmq.hpp>

#include "processor/flow_control/flow_control_entrance.h"
#include "utils/utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";
constexpr auto MC_INPUT_NAME = "Placeholder";
constexpr auto MC_OUTPUT_NAME = "probabilities:0";


#define HACK

ImageMatch::ImageMatch(unsigned int vishash_size, unsigned int batch_size)
    : Processor(PROCESSOR_TYPE_IMAGEMATCH, {SOURCE_NAME}, {SINK_NAME}),
      vishash_size_(vishash_size),
      batch_size_(batch_size) {}

std::shared_ptr<ImageMatch> ImageMatch::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void ImageMatch::AddQuery(const std::string& model_path, float threshold) {
  std::lock_guard<std::mutex> guard(query_guard_);
  int query_id = query_data_.size();
  query_t* current_query = &query_data_[query_id];
  current_query->matches = std::make_unique<Eigen::VectorXf>(batch_size_);
  current_query->query_id = query_id;
  current_query->threshold = threshold;
  SetClassifier(current_query, model_path);
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
  // Start time for benchmarking purposes
  auto start_time = boost::posix_time::microsec_clock::local_time();

  auto frame = GetFrame("input");
  CHECK(frame != nullptr);
  // If no queries, Send frame with empty imagematch fields
  std::lock_guard<std::mutex> guard(query_guard_);
  if (query_data_.empty()) {
    std::vector<int> image_match_matches;
    frame->SetValue("imagematch.matches", image_match_matches);
    frame->SetValue("imagematch.end_to_end_time_micros", 0);
    frame->SetValue("imagematch.matrix_multiply_time_micros", 0);
    PushFrame(SINK_NAME, std::move(frame));
    return;
  }
#ifndef HACK
  cv::Mat feature_vector = frame->GetValue<cv::Mat>("feature_vector");
#endif
  frames_batch_.push_back(std::move(frame));
  if (frames_batch_.size() < batch_size_) {
    return;
  }

  // Calculate similarity using Micro Classifiers
  auto overhead_end_time = boost::posix_time::microsec_clock::local_time();

#ifndef HACK
  int height = feature_vector.rows;
  int width = feature_vector.cols;
  int channel = feature_vector.channels();
#endif
#ifdef HACK
  int height = 1;
  int width = 1;
  int channel = 9216;
#endif
  tensorflow::Tensor input_tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({static_cast<long long>(batch_size_),
                                height, width, channel}));
  int count = 0;
  for (const auto& frame : frames_batch_) {
#ifndef HACK
    std::copy_n((float*)frame->GetValue<cv::Mat>("feature_vector").data,
                height * width * channel,
                input_tensor.flat<float>().data() + count++ * channel * height * width);
#endif
  }
  for (auto& query : query_data_) {
    std::vector<tensorflow::Tensor> outputs;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    inputs.push_back({MC_INPUT_NAME, input_tensor});
    tensorflow::Status status = query.second.classifier->Run(inputs, {MC_OUTPUT_NAME}, {}, &outputs);
   if (!status.ok()) {
      LOG(FATAL) << "Session::Run() completed with errors: "
                 << status.error_message();
  } 
  CHECK(outputs.size() == 1) << "Outputs should be of size 1, got " << outputs.size();;
  }

  auto matrix_end_time = boost::posix_time::microsec_clock::local_time();
  for (decltype(frames_batch_.size()) batch_idx = 0;
       batch_idx < frames_batch_.size(); ++batch_idx) {
    std::vector<int> image_match_matches;
    /*for (auto& query : query_data_) {
    // TODO fix this block
      if ((*(query.second.matches))(batch_idx) == 1) {
        image_match_matches.push_back(query.second.query_id);
      }
    }
    frames_batch_.at(batch_idx)->SetValue("imagematch.matches",
                                          image_match_matches);
  */
    frames_batch_.at(batch_idx)->SetValue("imagematch.matches",
                                          image_match_matches);
    auto end_time = boost::posix_time::microsec_clock::local_time();
    frames_batch_.at(batch_idx)->SetValue(
        "imagematch.end_to_end_time_micros",
        (end_time - start_time).total_microseconds());
    frames_batch_.at(batch_idx)->SetValue(
        "imagematch.matrix_multiply_time_micros",
        (matrix_end_time - overhead_end_time).total_microseconds());
    PushFrame(SINK_NAME, std::move(frames_batch_.at(batch_idx)));
  }
  frames_batch_.clear();
  return;
}

void ImageMatch::SetClassifier(query_t* current_query,
                               const std::string& model_path) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status status = ReadBinaryProto(
      tensorflow::Env::Default(), model_path, &graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to load TensorFlow graph: " << status.error_message();
  }

  current_query->classifier.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  status = current_query->classifier->Create(graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to create TensorFlow Session: "
               << status.error_message();
  }
}
