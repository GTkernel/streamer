/**
 * Multi-target detection using Caffe Yolo
 *
 * @author Wendy Chin <wendy.chin@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include "yolo_detector.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "utils/yolo_utils.h"

namespace yolo {
Detector::Detector(const string& model_file, const string& weights_file) {
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

  /* Load the network. */
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFromBinaryProto(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

std::vector<float> Detector::Detect(cv::Mat& img) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  int size = width * height;
  cv::Mat image_resized;
  cv::resize(img, image_resized, cv::Size(height, width));

  float* input_data = input_layer->mutable_cpu_data();
  int temp, idx;
  for (int i = 0; i < height; ++i) {
    uchar* pdata = image_resized.ptr<uchar>(i);
    for (int j = 0; j < width; ++j) {
      temp = 3 * j;
      idx = i * width + j;
      input_data[idx] = (pdata[temp + 2] / 127.5) - 1;
      input_data[idx + size] = (pdata[temp + 1] / 127.5) - 1;
      input_data[idx + 2 * size] = (pdata[temp + 0] / 127.5) - 1;
    }
  }

  net_->Forward();

  caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  std::vector<float> DetectionResult(begin, end);
  return DetectionResult;
}
}  // namespace yolo

YoloDetector::YoloDetector(const ModelDesc& model_desc,
                           float confidence_threshold, float idle_duration,
                           const std::set<std::string>& targets)
    : Processor(PROCESSOR_TYPE_YOLO_DETECTOR, {"input"}, {"output"}),
      model_desc_(model_desc),
      confidence_threshold_(confidence_threshold),
      idle_duration_(idle_duration),
      targets_(targets) {}

std::shared_ptr<YoloDetector> YoloDetector::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

bool YoloDetector::Init() {
  std::string model_file = model_desc_.GetModelDescPath();
  std::string weights_file = model_desc_.GetModelParamsPath();
  LOG(INFO) << "model_file: " << model_file;
  LOG(INFO) << "weights_file: " << weights_file;
  auto mean_colors = ModelManager::GetInstance().GetMeanColors();
  std::ostringstream mean_colors_stream;
  mean_colors_stream << mean_colors[0] << "," << mean_colors[1] << ","
                     << mean_colors[2];

  detector_.reset(new yolo::Detector(model_file, weights_file));
  std::string labelmap_file = model_desc_.GetLabelFilePath();
  voc_names_ = ReadVocNames(labelmap_file);

  LOG(INFO) << "YoloDetector initialized";
  return true;
}

bool YoloDetector::OnStop() { return true; }

void YoloDetector::Process() {
  Timer timer;
  timer.Start();

  auto frame = GetFrame("input");
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = now - last_detect_time_;
  if (diff.count() >= idle_duration_) {
    auto image = frame->GetValue<cv::Mat>("image");
    std::vector<float> DetectionOutput = detector_->Detect(image);
    std::vector<std::vector<int>> bboxs;
    float pro_obj[49][2];
    int idx_class[49];

    std::vector<std::vector<int>> bboxes =
        GetBoxes(DetectionOutput, &pro_obj[0][0], idx_class, bboxs,
                 confidence_threshold_, image);

    std::vector<std::vector<int>> filtered_res;
    auto original_img = frame->GetValue<cv::Mat>("original_image");
    CHECK(!original_img.empty());
    std::vector<Rect> bbs;
    std::vector<string> tags;
    std::vector<float> confidences;

    for (size_t i = 0; i < bboxes.size(); ++i) {
      const std::vector<int>& d = bboxes[i];
      if (targets_.empty()) {
        filtered_res.push_back(d);
      } else {
        auto it = targets_.find(voc_names_.at(d[0]));
        if (it != targets_.end()) filtered_res.push_back(d);
      }
    }

    for (size_t i = 0; i < filtered_res.size(); ++i) {
      int xmin = filtered_res[i][1];
      int ymin = filtered_res[i][2];
      int xmax = filtered_res[i][3];
      int ymax = filtered_res[i][4];
      if (xmin < 0) xmin = 0;
      if (ymin < 0) ymin = 0;
      if (xmax > original_img.cols) xmax = original_img.cols;
      if (ymax > original_img.rows) ymax = original_img.rows;
      
      tags.push_back(voc_names_.at(filtered_res[i][0]));
      bbs.push_back(Rect(xmin, ymin,
                         xmax - xmin,
                         ymax - ymin));
      confidences.push_back(filtered_res[i][5] * 1.0 / 100);
    }

    last_detect_time_ = std::chrono::system_clock::now();
    frame->SetValue("tags", tags);
    frame->SetValue("bounding_boxes", bbs);
    frame->SetValue("confidences", confidences);
    PushFrame("output", std::move(frame));
    LOG(INFO) << "Yolo detection took " << timer.ElapsedMSec() << " ms";
  } else {
    PushFrame("output", std::move(frame));
  }
}
