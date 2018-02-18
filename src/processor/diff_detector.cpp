
#include "processor/diff_detector.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "utils/utils.h"

constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

DiffDetector::DiffDetector(double threshold, int block_size,
                           const std::string& weights_path,
                           const std::string& ref_path)
    : Processor(PROCESSOR_TYPE_DIFF_DETECTOR, {SOURCE_NAME}, {SINK_NAME}),
      threshold_(threshold),
      blocked_(true),
      block_size_(block_size),
      weights_(LoadWeights(weights_path)),
      dynamic_ref_(false),
      ref_img_(ReadRefImg(ref_path)) {
  LOG(INFO)
      << "DiffDetector configured with blocking and a static reference image.";
}

DiffDetector::DiffDetector(double threshold, int block_size,
                           const std::string& weights_path,
                           unsigned long t_diff_frames)
    : Processor(PROCESSOR_TYPE_DIFF_DETECTOR, {SOURCE_NAME}, {SINK_NAME}),
      threshold_(threshold),
      blocked_(true),
      block_size_(block_size),
      weights_(LoadWeights(weights_path)),
      dynamic_ref_(true),
      t_diff_frames_(t_diff_frames),
      buffer_{t_diff_frames} {
  LOG(INFO)
      << "DiffDetector configured with blocking and dynamic reference images.";
}

DiffDetector::DiffDetector(double threshold, const std::string& ref_path)
    : Processor(PROCESSOR_TYPE_DIFF_DETECTOR, {SOURCE_NAME}, {SINK_NAME}),
      threshold_(threshold),
      blocked_(false),
      dynamic_ref_(false),
      ref_img_(ReadRefImg(ref_path)) {
  LOG(INFO) << "DiffDetector configured with global scope and a static "
               "reference image.";
}

DiffDetector::DiffDetector(double threshold, unsigned long t_diff_frames)
    : Processor(PROCESSOR_TYPE_DIFF_DETECTOR, {SOURCE_NAME}, {SINK_NAME}),
      threshold_(threshold),
      blocked_(false),
      dynamic_ref_(true),
      t_diff_frames_(t_diff_frames),
      buffer_{t_diff_frames} {
  LOG(INFO) << "DiffDetector configured with global scope and dynamic "
               "reference images.";
}

std::shared_ptr<DiffDetector> DiffDetector::Create(const FactoryParamsType&) {
  STREAMER_NOT_IMPLEMENTED;
  return nullptr;
}

void DiffDetector::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE_NAME, stream);
}

StreamPtr DiffDetector::GetSink() { return Processor::GetSink(SINK_NAME); }

void DiffDetector::EnableLog(std::string output_dir) {
  std::string mode;
  if (blocked_) {
    mode = "blocked";
  } else {
    mode = "global";
  }

  std::string ref_type;
  if (dynamic_ref_) {
    ref_type = "dynamic";
  } else {
    ref_type = "static";
  }

  std::ostringstream filepath;
  filepath << output_dir << "/diff_detector_" << mode << "_" << ref_type << "_";
  if (dynamic_ref_) {
    filepath << t_diff_frames_ << "_";
  }
  filepath << threshold_ << ".log";
  log_.open(filepath.str());
}

bool DiffDetector::Init() { return true; }

bool DiffDetector::OnStop() { return true; }

void DiffDetector::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  auto img = frame->GetValue<cv::Mat>("image");
  auto frame_id = frame->GetValue<unsigned long>("frame_id");

  unsigned long ref_id;
  cv::Mat ref_img;
  if (dynamic_ref_) {
    if (!buffer_.size()) {
      // If the buffer is empty, then this is the first frame. It should not be
      // dropped, since we do not have a baseline yet.
      buffer_.push_back(std::make_pair(frame_id, img.clone()));
      PushFrame(SINK_NAME, std::move(frame));
      return;
    } else {
      // If this is not the first frame, then the head of the buffer is our
      // reference image. This code works even if the buffer has not filled up
      // yet, in which case the reference image will be the first frame.
      std::pair<unsigned long, cv::Mat> ref_pair = buffer_.front();
      ref_id = ref_pair.first;
      ref_img = ref_pair.second;
      buffer_.push_back(std::make_pair(frame_id, img.clone()));
    }
  } else {
    ref_img = ref_img_;
  }

  int img_num_channels = img.channels();
  cv::Size img_size = img.size();
  auto img_width = img_size.width;
  auto img_height = img_size.height;

  int ref_img_num_channels = ref_img.channels();
  cv::Size ref_img_size = ref_img.size();
  auto ref_img_width = ref_img_size.width;
  auto ref_img_height = ref_img_size.height;

  CHECK(ref_img_num_channels == img_num_channels)
      << "The reference frame and the current frame have different numbers "
      << "of channels (" << ref_img_num_channels << " vs. " << img_num_channels
      << ")";
  CHECK(ref_img_width == img_width)
      << "The reference frame and the current frame have different widths ("
      << ref_img_width << " vs. " << img_width << ")";
  CHECK(ref_img_height == img_height)
      << "The reference frame and the current frame have different heights ("
      << ref_img_height << " vs. " << img_height << ")";
  CHECK(img.type() == ref_img.type())
      << "The reference frame and the current frame are different types!";

  double diff = 0.0;
  auto start_time_micros = boost::posix_time::microsec_clock::local_time();
  if (blocked_) {
    diff = BlockedMse(img, ref_img, weights_, block_size_);
  } else {
    diff = GlobalMse(img, ref_img);
  }
  auto diff_micros =
      boost::posix_time::microsec_clock::local_time() - start_time_micros;
  frame->SetValue("DiffDetector.diff_micros", diff_micros);

  if (diff > threshold_) {
    if (log_.is_open()) {
      log_ << frame->GetValue<unsigned long>("frame_id") << std::endl;
    }
    PushFrame(SINK_NAME, std::move(frame));
  }
}

double DiffDetector::GlobalMse(cv::Mat img, cv::Mat ref_img) {
  // This function is based heavily on the NOSCOPE
  // "noscope::filters::GlobalMSE()" function defined here:
  // https://github.com/stanford-futuredata/tensorflow-noscope/blob/25abedd53e2b20fe27cab05978bf847b930f8211/tensorflow/noscope/filters.cc
  cv::Mat diff;
  cv::absdiff(img, ref_img, diff);
  cv::Mat mult;
  cv::multiply(diff, diff, mult);
  auto sum = cv::sum(cv::sum(mult))[0];
  return sum / (double)(img.channels() * img.total());
}

double DiffDetector::BlockedMse(cv::Mat img, cv::Mat ref_img, cv::Mat weights,
                                int block_size) {
  // The remainder of this function is based heavily on the NOSCOPE
  // "noscope::filters::BlockedMSE()" function defined here:
  // https://github.com/stanford-futuredata/tensorflow-noscope/blob/25abedd53e2b20fe27cab05978bf847b930f8211/tensorflow/noscope/filters.cc

  LOG(INFO) << "Running blocked MSE...";

  cv::Mat diff;  // (size, type);
  cv::absdiff(img, ref_img, diff);
  diff.convertTo(diff, CV_32FC3);
  cv::Mat mult;
  cv::multiply(diff, diff, mult);

  std::vector<cv::Mat> channels;
  cv::split(mult, channels);
  auto num_extracted_channels = channels.size();

  int num_channels = img.channels();
  cv::Size size = mult.size();
  auto width = size.width;
  auto height = size.height;
  int num_blocks = width / block_size;

  int weights_channels = weights.channels();
  cv::Size weights_size = weights.size();
  auto weights_width = weights_size.width;
  auto weights_height = weights_size.height;

  CHECK(num_extracted_channels == (unsigned long)num_channels)
      << "Image has " << num_channels << " channels, but we only extracted "
      << num_extracted_channels << " channels.";
  CHECK(width == height)
      << "The BlockedMse() difference detector only supports square images!";
  CHECK(weights_channels == num_channels)
      << "The weights matrix has " << weights_channels
      << " channels, but the image has " << num_channels << "!";
  CHECK(weights_width == num_blocks)
      << "The weights matrix has a width of " << weights_width
      << ", but the image is " << num_blocks << " blocks wide!";
  CHECK(weights_height == num_blocks)
      << "The weights matrix has a height of " << weights_height
      << ", but the image is " << num_blocks << " blocks tall!";

  // TODO: Finish debugging this code. There is something terribly wrong with
  //       we're using cv::Mats.
  double mse = 0;
  for (decltype(num_extracted_channels) i = 0; i < num_extracted_channels;
       ++i) {
    for (decltype(num_blocks) j = 0; j < num_blocks; ++j) {
      for (decltype(num_blocks) k = 0; k < num_blocks; ++k) {
        cv::Rect block(cv::Point(j * block_size, k * block_size),
                       cv::Point((j + 1) * block_size, (k + 1) * block_size));
        cv::Mat tmp = channels[i](block);
        mse += cv::sum(tmp)[0] * weights.at<double>(i, j, k);
      }
    }
  }
  return mse;
}

cv::Mat DiffDetector::LoadWeights(const std::string& weights_path) {
  std::ifstream weights_f(weights_path);
  if (!weights_f.good()) {
    std::ostringstream msg;
    msg << "Unable to read weights file: " << weights_path;
    throw std::runtime_error(msg.str());
  }
  // Read the entire file into a string.
  std::string weights_str((std::istreambuf_iterator<char>(weights_f)),
                          std::istreambuf_iterator<char>());
  // Wrap the string in a stream.
  std::stringstream stream(weights_str);

  cv::Mat weights;
  try {
    boost::archive::binary_iarchive ar(stream);
    ar >> weights;
  } catch (const boost::archive::archive_exception& e) {
    LOG(ERROR) << "Boost serialization error: " << e.what();
  }
  return weights;
}

cv::Mat DiffDetector::ReadRefImg(const std::string& ref_path) {
  cv::Mat img = cv::imread(ref_path, cv::IMREAD_COLOR);
  if (img.data == NULL) {
    std::ostringstream msg;
    msg << "Unable to read reference image: " << ref_path;
    throw std::runtime_error(msg.str());
  }

  // Convert to float.
  cv::Mat img_converted;
  if (img.channels() == 3) {
    img.convertTo(img_converted, CV_32FC3);
  } else {
    img.convertTo(img_converted, CV_32FC1);
  }
  return img_converted;
}
