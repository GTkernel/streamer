/**
 * Multi-target detection using FRCNN
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include "object_detector.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "common/context.h"
#include "model/model_manager.h"

ObjectDetector::ObjectDetector(const ModelDesc& model_desc, Shape input_shape,
                               float idle_duration,
                               const std::set<std::string>& targets)
    : Processor(PROCESSOR_TYPE_OBJECT_DETECTOR, {"input"}, {"output"}),
      model_desc_(model_desc),
      input_shape_(input_shape),
      idle_duration_(idle_duration),
      targets_(targets) {}

bool ObjectDetector::Init() {
  // Set Caffe backend
  int desired_device_number = Context::GetContext().GetInt(DEVICE_NUMBER);

  if (desired_device_number == DEVICE_NUMBER_CPU_ONLY) {
    LOG(INFO) << "Use device: " << desired_device_number << "(CPU)";
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  } else {
#ifdef USE_CUDA
    std::vector<int> gpus;
    GetCUDAGpus(gpus);

    if (desired_device_number < gpus.size()) {
      // Device exists
      LOG(INFO) << "Use GPU with device ID " << desired_device_number;
      caffe::Caffe::SetDevice(desired_device_number);
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
    } else {
      LOG(FATAL) << "No GPU device: " << desired_device_number;
    }
#elif USE_OPENCL
    std::vector<int> gpus;
    int count = caffe::Caffe::EnumerateDevices();

    if (desired_device_number < count) {
      // Device exists
      LOG(INFO) << "Use GPU with device ID " << desired_device_number;
      caffe::Caffe::SetDevice(desired_device_number);
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
    } else {
      LOG(FATAL) << "No GPU device: " << desired_device_number;
    }
#else
    LOG(FATAL) << "Compiled in CPU_ONLY mode but have a device number "
                  "configured rather than -1";
#endif  // USE_CUDA
  }

  std::string proto_file = model_desc_.GetModelDescPath();
  std::string model_file = model_desc_.GetModelParamsPath();
  std::string voc_config = model_desc_.GetVocConfigPath();
  LOG(INFO) << "proto_file: " << proto_file;
  LOG(INFO) << "model_file: " << model_file;
  LOG(INFO) << "voc_config: " << voc_config;

  API::Set_Config(voc_config);
  detector_.reset(new API::Detector(proto_file, model_file));

  LOG(INFO) << "ObjectDetector initialized";
  return true;
}

bool ObjectDetector::OnStop() { return true; }

void ObjectDetector::Process() {
  Timer timer;
  timer.Start();

  auto frame = GetFrame("input");
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = now - last_detect_time_;
  if (diff.count() >= idle_duration_) {
    auto image = frame->GetValue<cv::Mat>("image");
    CHECK(image.channels() == input_shape_.channel &&
          image.size[1] == input_shape_.width &&
          image.size[0] == input_shape_.height);
    std::vector<caffe::Frcnn::BBox<float>> results;
    detector_->predict(image, results);
    /*
    LOG(INFO) << "There are " << results.size() << " objects in picture.";
    for (size_t obj = 0; obj < results.size(); obj++) {
      LOG(INFO) << results[obj].to_string() << "\t\t"
                <<
    caffe::Frcnn::GetClassName(caffe::Frcnn::LoadVocClass(),results[obj].id);
    }
    */

    std::vector<caffe::Frcnn::BBox<float>> filtered_res;
    for (const auto& m : results) {
      if (m.confidence > 0.5) {
        if (targets_.empty()) {
          filtered_res.push_back(m);
        } else {
          auto it = targets_.find(
              caffe::Frcnn::GetClassName(caffe::Frcnn::LoadVocClass(), m.id));
          if (it != targets_.end()) filtered_res.push_back(m);
        }
      }
    }

    auto original_img = frame->GetValue<cv::Mat>("original_image");
    CHECK(!original_img.empty());
    std::vector<string> tags;
    std::vector<Rect> bboxes;
    std::vector<float> confidences;
    float scale_factor[] = {(float)original_img.size[1] / (float)image.size[1],
                            (float)original_img.size[0] / (float)image.size[0]};
    for (const auto& m : filtered_res) {
      tags.push_back(
          caffe::Frcnn::GetClassName(caffe::Frcnn::LoadVocClass(), m.id));
      bboxes.push_back(Rect((int)(m[0] * scale_factor[0]),
                            (int)(m[1] * scale_factor[1]),
                            (int)((m[2] - m[0]) * scale_factor[0]),
                            (int)((m[3] - m[1]) * scale_factor[1])));
      confidences.push_back(m.confidence);
    }

    last_detect_time_ = std::chrono::system_clock::now();
    frame->SetValue("tags", tags);
    frame->SetValue("bounding_boxes", bboxes);
    frame->SetValue("confidences", confidences);
    PushFrame("output", std::move(frame));
    LOG(INFO) << "Object detection took " << timer.ElapsedMSec() << " ms";
  } else {
    PushFrame("output", std::move(frame));
  }
}
