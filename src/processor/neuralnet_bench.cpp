
#include "neuralnet_bench.h"

#include "model/model_manager.h"
#include "utils/string_utils.h"
#include "utils/utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";
#define LAYER "dense_2/Sigmoid:0"
//#define LAYER "prob"

NNBench::NNBench(const ModelDesc& model_desc, const Shape& input_shape,
                 size_t batch_size, int num_classifiers, bool run)
    : Processor(PROCESSOR_TYPE_NEURAL_NET_EVALUATOR, {SOURCE_NAME},
                {SINK_NAME}),
      input_shape_(input_shape),
      batch_size_(batch_size),
      classifiers_(num_classifiers),
      run_(run) {
  // Load model.
  auto& manager = ModelManager::GetInstance();
  for (int i = 0; i < num_classifiers; ++i) {
    auto model = manager.CreateModel(model_desc, input_shape_, batch_size_);
    model->Load();
    models_.push_back(std::move(model));
  }
}

NNBench::~NNBench() {}

const std::vector<std::string> NNBench::GetSinkNames() const {
  STREAMER_NOT_IMPLEMENTED;
  return {};
}

std::shared_ptr<NNBench> NNBench::Create(const FactoryParamsType& params) {
  return nullptr;
}

bool NNBench::Init() { return true; }

bool NNBench::OnStop() { return true; }

void NNBench::SetSource(const std::string& name, StreamPtr stream,
                        const std::string& layername) {
  if (layername == "") {
    input_layer_name_ = models_.at(0)->GetModelDesc().GetDefaultInputLayer();
  } else {
    input_layer_name_ = layername;
  }
  LOG(INFO) << "Using layer \"" << input_layer_name_
            << "\" as input for source \"" << name << "\"";
  Processor::SetSource(name, stream);
}

void NNBench::SetSource(StreamPtr stream, const std::string& layername) {
  SetSource(SOURCE_NAME, stream, layername);
}

StreamPtr NNBench::GetSink() { return Processor::GetSink(SINK_NAME); }

void NNBench::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  auto enter_time = boost::posix_time::microsec_clock::local_time();
  cv::Mat input_mat;
  cur_batch_frames_.push_back(std::move(input_frame));
  if (cur_batch_frames_.size() < batch_size_) {
    return;
  }
  std::vector<cv::Mat> cur_batch_;
  for (auto& frame : cur_batch_frames_) {
    // NOTE: this is broken now
    if (frame->Count("activations") > 0) {
      cur_batch_.push_back(frame->GetValue<cv::Mat>("activations"));
    } else {
      cur_batch_.push_back(frame->GetValue<cv::Mat>("image"));
    }
  }

  auto start_time = boost::posix_time::microsec_clock::local_time();
  std::vector<std::thread> threads;
  std::map<std::string, std::vector<cv::Mat>> input_map;
  input_map[input_layer_name_] = cur_batch_;
  CHECK(models_.size() == classifiers_);
  for (int i = 0; i < models_.size(); ++i) {
    models_.at(i)->Evaluate({{input_layer_name_, cur_batch_}}, {LAYER},
                            nullptr);
  }
  long time_elapsed =
      (boost::posix_time::microsec_clock::local_time() - start_time)
          .total_microseconds();

  // Push the activations for each published layer to their respective sink.
  for (decltype(cur_batch_frames_.size()) i = 0; i < cur_batch_frames_.size();
       ++i) {
    std::unique_ptr<Frame> ret_frame = std::move(cur_batch_frames_.at(i));
    ret_frame->SetValue("thetime", time_elapsed);
    PushFrame(SINK_NAME, std::move(ret_frame));
  }
  cur_batch_frames_.clear();
}
