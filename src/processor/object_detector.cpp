#include "common/context.h"
#include "object_detector.h"
#include "model/model_manager.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"

#define GET_SOURCE_NAME(i) ("input" + std::to_string(i))
#define GET_SINK_NAME(i) ("output" + std::to_string(i))

ObjectDetector::ObjectDetector(const ModelDesc &model_desc,
                               Shape input_shape,
                               size_t batch_size)
    : Processor({}, {}),
      model_desc_(model_desc),
      input_shape_(input_shape),
      batch_size_(batch_size) {
  for (size_t i = 0; i < batch_size_; i++) {
    sources_.insert({"input" + std::to_string(i), nullptr});
    sinks_.insert({"output" + std::to_string(i),
                   std::shared_ptr<Stream>(new Stream)});
  }

  LOG(INFO) << "batch size of " << batch_size_;
}

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
#endif
  }

  auto &manager = ModelManager::GetInstance();
  auto mean_colors = manager.GetMeanColors();
  mean_image_ = cv::Mat(cv::Size(input_shape_.width, input_shape_.height),
          CV_32FC3,
          cv::Scalar(mean_colors[0], mean_colors[1], mean_colors[2]));

  input_buffer_ =
      DataBuffer(batch_size_ * input_shape_.GetSize() * sizeof(float));

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

bool ObjectDetector::OnStop() {
  model_.reset(nullptr);
  return true;
}

void ObjectDetector::Process() {
  Timer timer;
  timer.Start();

  std::vector<caffe::Frcnn::BBox<float> > results;
  for (int i = 0; i < batch_size_; i++) {
    auto image_frame = GetFrame<ImageFrame>(GET_SOURCE_NAME(i));
    cv::Mat img = image_frame->GetImage();
    CHECK(img.channels() == input_shape_.channel &&
            img.size[0] == input_shape_.width &&
            img.size[1] == input_shape_.height);
    detector_->predict(img, results);
    LOG(INFO) << "There are " << results.size() << " objects in picture.";
    for (size_t obj = 0; obj < results.size(); obj++) {
      LOG(INFO) << results[obj].to_string() << "\t\t"
                << caffe::Frcnn::GetClassName(caffe::Frcnn::LoadVocClass(),results[obj].id);
    }

    std::vector<caffe::Frcnn::BBox<float> > filtered_res;
    for (const auto& m: results) {
      if (m.confidence > 0.5) {
        filtered_res.push_back(m);
      }
    }

    cv::Mat original_img = image_frame->GetOriginalImage();
    CHECK(!original_img.empty());
    std::vector<string> tags;
    std::vector<Rect> bboxes;
    float factor = (float)original_img.size[0]/(float)img.size[0];
    float factor1 = ((float)original_img.size[1]-(float)original_img.size[0])/2.0;
    for (const auto& m: filtered_res) {
      std::ostringstream text;
      text << caffe::Frcnn::GetClassName(caffe::Frcnn::LoadVocClass(), m.id) << "  :  " << m.confidence;
      tags.push_back(text.str());
      bboxes.push_back(Rect((int)((m[0]*factor)+factor1), (int)(m[1]*factor), (int)((m[2]-m[0])*factor), (int)((m[3]-m[1])*factor)));
    }
    PushFrame(GET_SINK_NAME(i), new MetadataFrame(tags, bboxes, original_img));
  }

  LOG(INFO) << "Object detection took " << timer.ElapsedMSec() << " ms";
}

ProcessorType ObjectDetector::GetType() {
  return PROCESSOR_TYPE_OBJECT_DETECTOR;
}

void ObjectDetector::SetInputStream(int src_id, StreamPtr stream) {
  SetSource(GET_SOURCE_NAME(src_id), stream);
}
