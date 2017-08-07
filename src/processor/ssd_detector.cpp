/**
 * Multi-target detection using SSD
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include "ssd_detector.h"
#include "common/context.h"
#include "model/model_manager.h"

namespace ssd {
Detector::Detector(const string& model_file, const string& weights_file,
                   const string& mean_file, const string& mean_value) {
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

  /* Load the network. */
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<std::vector<float> > Detector::Detect(const cv::Mat& img) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_, input_geometry_.height,
                       input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  caffe::Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  std::vector<std::vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    std::vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty())
        << "Cannot specify mean_file and mean_value at the same time";
    caffe::BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    caffe::Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (decltype(num_channels_) i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty())
        << "Cannot specify mean_file and mean_value at the same time";
    std::stringstream ss(mean_value);
    std::vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_)
        << "Specify either 1 mean_value or as many as channels: "
        << num_channels_;

    std::vector<cv::Mat> channels;
    for (decltype(num_channels_) i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                          std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data) ==
        net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}
}  // namespace ssd

SsdDetector::SsdDetector(const ModelDesc& model_desc, Shape input_shape,
                         float confidence_threshold, float idle_duration,
                         const std::set<std::string>& targets)
    : Processor(PROCESSOR_TYPE_SSD_DETECTOR, {"input"}, {"output"}),
      model_desc_(model_desc),
      input_shape_(input_shape),
      confidence_threshold_(confidence_threshold),
      idle_duration_(idle_duration),
      targets_(targets) {}

std::shared_ptr<SsdDetector> SsdDetector::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

bool SsdDetector::Init() {
  std::string model_file = model_desc_.GetModelDescPath();
  std::string weights_file = model_desc_.GetModelParamsPath();
  LOG(INFO) << "model_file: " << model_file;
  LOG(INFO) << "weights_file: " << weights_file;
  auto mean_colors = ModelManager::GetInstance().GetMeanColors();
  std::ostringstream mean_colors_stream;
  mean_colors_stream << mean_colors[0] << "," << mean_colors[1] << ","
                     << mean_colors[2];

  detector_.reset(new ssd::Detector(model_file, weights_file, "",
                                    mean_colors_stream.str()));

  std::string labelmap_file = model_desc_.GetLabelFilePath();
  CHECK(ReadProtoFromTextFile(labelmap_file, &label_map_))
      << "Failed to parse LabelMap file: " << labelmap_file;

  LOG(INFO) << "SsdDetector initialized";
  return true;
}

bool SsdDetector::OnStop() { return true; }

void SsdDetector::Process() {
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
    std::vector<std::vector<float> > detections = detector_->Detect(image);
    std::vector<std::vector<float> > filtered_res;
    for (decltype(detections.size()) i = 0; i < detections.size(); ++i) {
      const std::vector<float>& d = detections[i];
      // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
      CHECK_EQ(d.size(), 7);
      const float score = d[2];
      if (score >= confidence_threshold_) {
        if (targets_.empty()) {
          filtered_res.push_back(d);
        } else {
          auto it = targets_.find(GetLabelName((int)d[1]));
          if (it != targets_.end()) filtered_res.push_back(d);
        }
      }
    }

    auto original_img = frame->GetValue<cv::Mat>("original_image");
    CHECK(!original_img.empty());
    std::vector<string> tags;
    std::vector<Rect> bboxes;
    std::vector<float> confidences;
    for (const auto& m : filtered_res) {
      int xmin = m[3] * original_img.cols;
      int ymin = m[4] * original_img.rows;
      int xmax = m[5] * original_img.cols;
      int ymax = m[6] * original_img.rows;
      if (xmin < 0) xmin = 0;
      if (ymin < 0) ymin = 0;
      if (xmax > original_img.cols) xmax = original_img.cols;
      if (ymax > original_img.rows) ymax = original_img.rows;

      tags.push_back(GetLabelName((int)m[1]));
      bboxes.push_back(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
      confidences.push_back(m[2]);
    }

    last_detect_time_ = std::chrono::system_clock::now();
    frame->SetValue("tags", tags);
    frame->SetValue("bounding_boxes", bboxes);
    frame->SetValue("confidences", confidences);
    PushFrame("output", std::move(frame));
    LOG(INFO) << "Ssd detection took " << timer.ElapsedMSec() << " ms";
  } else {
    PushFrame("output", std::move(frame));
  }
}

std::string SsdDetector::GetLabelName(int label) const {
  std::string name;
  int item_size = label_map_.item_size();
  for (int i = 0; i < item_size; ++i) {
    auto item = label_map_.item(i);
    if (item.label() == label) {
      name = item.name();
    }
  }

  CHECK(!name.empty()) << "Cannot find a label name";
  return name;
}
