
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

ImageMatch::ImageMatch(unsigned int vishash_size, unsigned int batch_size)
    : Processor(PROCESSOR_TYPE_IMAGEMATCH, {SOURCE_NAME}, {SINK_NAME}),
      vishash_size_(vishash_size),
      batch_size_(batch_size) {}

std::shared_ptr<ImageMatch> ImageMatch::Create(const FactoryParamsType&) {
  // TODO
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void ImageMatch::AddQuery(const std::string& model_path,
                          const std::string& params_path, float threshold) {
  std::lock_guard<std::mutex> guard(query_guard_);
  int query_id = query_data_.size();
  query_t* current_query = &query_data_[query_id];
  current_query->matches = std::make_unique<Eigen::VectorXf>(batch_size_);
  current_query->query_id = query_id;
  current_query->threshold = threshold;
  SetClassifier(current_query, model_path, params_path);
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
  // TODO: Send to exit instead of releasing frame
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
  frames_batch_.push_back(std::move(frame));
  if (frames_batch_.size() < batch_size_) {
    return;
  }

  // Calculate similarity using Micro Classifiers
  std::lock_guard<std::mutex> guard(query_guard_);
  auto overhead_end_time = boost::posix_time::microsec_clock::local_time();

  for (auto& query : query_data_) {
    float* data =
        query.second.classifier->input_blobs().at(0)->mutable_cpu_data();
    for (const auto& frame : frames_batch_) {
      memcpy(data, frame->GetValue<cv::Mat>("activations").data,
             vishash_size_ * sizeof(float));
      data += vishash_size_;
    }
    query.second.classifier->Forward();
    // TODO Add error message
    CHECK_EQ(query.second.classifier->output_blobs().size(), 1);
    caffe::Blob<float>* output_layer =
        query.second.classifier->output_blobs()[0];
    auto layer_outputs = query.second.classifier->top_vecs();
    for (decltype(batch_size_) i = 0; i < batch_size_; ++i) {
      // Get probability of a match on this Micro Classifier
      // The expected output is the logit exponent (pre-softmax)
      // TODO: get softmax instead
      // The output is expected to be of the format
      //                2
      // b [ logit_nomatch logit_match ]
      // a [ . .                       ]
      // t [ .        .                ]
      // c [ .               .         ]
      // h [ .                      .  ]
      float p_nomatch = output_layer->cpu_data()[i * 2];
      float p_match = output_layer->cpu_data()[i * 2 + 1];
      // Do softmax
      //   exponentiate
      p_nomatch = std::exp(p_nomatch);
      p_match = std::exp(p_match);
      float total = p_match + p_nomatch;
      //   normalize by total
      p_nomatch /= total;
      p_match /= total;
      if (p_match > query.second.threshold) {
        ((*query.second.matches))(i) = 1;
      } else {
        ((*query.second.matches))(i) = 0;
      }
    }
  }

  auto matrix_end_time = boost::posix_time::microsec_clock::local_time();
  for (decltype(frames_batch_.size()) batch_idx = 0;
       batch_idx < frames_batch_.size(); ++batch_idx) {
    std::vector<int> image_match_matches;
    for (auto& query : query_data_) {
      if ((*(query.second.matches))(batch_idx) == 1) {
        image_match_matches.push_back(query.second.query_id);
      }
    }
    frames_batch_.at(batch_idx)->SetValue("imagematch.matches",
                                          image_match_matches);
    auto end_time = boost::posix_time::microsec_clock::local_time();
    frames_batch_.at(batch_idx)->SetValue(
        "imagematch.end_to_end_time_micros",
        (end_time - start_time).total_microseconds());
    frames_batch_.at(batch_idx)->SetValue(
        "imagematch.matrix_multiply_time_micros",
        (matrix_end_time - overhead_end_time).total_microseconds());
    PushFrame("output", std::move(frames_batch_.at(batch_idx)));
  }
  frames_batch_.clear();
  return;
}

void ImageMatch::SetClassifier(query_t* current_query,
                               const std::string& model_path,
                               const std::string& params_path) {
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  current_query->classifier =
      std::make_unique<caffe::Net<float>>(model_path, caffe::TEST);
  current_query->classifier->CopyTrainedLayersFrom(params_path);
  CHECK_EQ(current_query->classifier->num_inputs(), 1);
  CHECK_EQ(current_query->classifier->num_outputs(), 1);
  caffe::Blob<float>* input_layer =
      current_query->classifier->input_blobs().at(0);
  input_layer->Reshape(batch_size_, 1, 1, vishash_size_);
  current_query->classifier->Reshape();
}
