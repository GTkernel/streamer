
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

ImageMatch::ImageMatch(unsigned int batch_size)
    : Processor(PROCESSOR_TYPE_IMAGEMATCH, {SOURCE_NAME}, {SINK_NAME}),
      batch_size_(batch_size) {}

std::shared_ptr<ImageMatch> ImageMatch::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void ImageMatch::AddQuery(const std::string& model_path, std::string layer_name,
                          float threshold, int xmin, int ymin, int xmax,
                          int ymax, bool flat) {
  std::lock_guard<std::mutex> guard(query_guard_);
  int query_id = query_data_.size();
  query_t* current_query = &query_data_[query_id];
  current_query->query_id = query_id;
  current_query->threshold = threshold;
  current_query->fv_spec = FvSpec(layer_name, xmin, ymin, xmax, ymax, flat);
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
  auto frame = GetFrame("input");
  CHECK(frame != nullptr);
  // If no queries, Send frame with empty imagematch fields
  std::lock_guard<std::mutex> guard(query_guard_);
  if (query_data_.empty()) {
    std::vector<int> image_match_matches;
    frame->SetValue("ImageMatch.matches", image_match_matches);
    PushFrame(SINK_NAME, std::move(frame));
    return;
  }
  frames_batch_.push_back(std::move(frame));
  if (frames_batch_.size() < batch_size_) {
    return;
  }
  // Calculate similarity using Micro Classifiers
  for (auto& query : query_data_) {
    cv::Mat fv = frames_batch_.at(0)->GetValue<cv::Mat>(
        FvSpec::GetUniqueID(query.second.fv_spec));
    int height = fv.rows;
    int width = fv.cols;
    int channel = fv.channels();
    tensorflow::Tensor input_tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape(
            {static_cast<long long>(batch_size_), height, width, channel}));
    int cur_batch_idx = 0;
    for (const auto& frame : frames_batch_) {
      cv::Mat fv =
          frame->GetValue<cv::Mat>(FvSpec::GetUniqueID(query.second.fv_spec));
      for (int row = 0; row < height; ++row) {
        std::copy_n(fv.ptr<float>(row), width * channel,
                    input_tensor.flat<float>().data() +
                        cur_batch_idx * height * width * channel +
                        row * channel * width);
      }
      cur_batch_idx += 1;
    }
    std::vector<tensorflow::Tensor> outputs;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    inputs.push_back({MC_INPUT_NAME, input_tensor});
    tensorflow::Status status =
        query.second.classifier->Run(inputs, {MC_OUTPUT_NAME}, {}, &outputs);
    if (!status.ok()) {
      LOG(FATAL) << "Session::Run() completed with errors: "
                 << status.error_message();
    }
    CHECK(outputs.size() == 1)
        << "Outputs should be of size 1, got " << outputs.size();
    ;
    const auto& output_tensor = outputs.at(0);
    int cur_dim = (*output_tensor.shape().begin()).size;
    for (int i = 0; i < cur_dim * 2; i += 2) {
      float prob_match = output_tensor.flat<float>().data()[i];
      // LOG(INFO) << prob_match;
      if (prob_match > query.second.threshold) {
        query.second.matches.push_back(1);
      } else {
        query.second.matches.push_back(0);
      }
    }
  }

  for (decltype(frames_batch_.size()) batch_idx = 0;
       batch_idx < frames_batch_.size(); ++batch_idx) {
    std::vector<int> image_match_matches;
    for (auto& query : query_data_) {
      if (query.second.matches.at(batch_idx) == 1) {
        image_match_matches.push_back(query.second.query_id);
      }
    }
    frames_batch_.at(batch_idx)->SetValue("ImageMatch.matches",
                                          image_match_matches);
    PushFrame(SINK_NAME, std::move(frames_batch_.at(batch_idx)));
  }
  frames_batch_.clear();
  return;
}

void ImageMatch::SetClassifier(query_t* current_query,
                               const std::string& model_path) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status status =
      ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to load TensorFlow graph: " << status.error_message();
  }

  current_query->classifier.reset(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  status = current_query->classifier->Create(graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to create TensorFlow Session: "
               << status.error_message();
  }
}
