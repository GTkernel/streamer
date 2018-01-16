
#ifndef STREAMER_PROCESSOR_DIFFERENCE_DETECTOR_H_
#define STREAMER_PROCESSOR_DIFFERENCE_DETECTOR_H_

#include <memory>
#include <string>
#include <utility>

#include <boost/circular_buffer.hpp>
#include <opencv2/opencv.hpp>

#include "common/types.h"
#include "processor/processor.h"
#include "stream/frame.h"

class DiffDetector : public Processor {
 public:
  DiffDetector(double threshold, int block_size,
               const std::string& weights_path, const std::string& ref_path);
  DiffDetector(double threshold, int block_size,
               const std::string& weights_path, unsigned long t_diff_frames);
  DiffDetector(double threshold, const std::string& ref_path);
  DiffDetector(double threshold, unsigned long t_diff_frames);

  static std::shared_ptr<DiffDetector> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  static double GlobalMse(cv::Mat img, cv::Mat ref_img);
  static double BlockedMse(cv::Mat img, cv::Mat ref_img, cv::Mat weights,
                           int block_size);
  static cv::Mat LoadWeights(const std::string& weights_path);
  static cv::Mat ReadRefImg(const std::string& ref_path);

  double threshold_;

  bool blocked_;
  int block_size_;
  cv::Mat weights_;

  // Indicates whether the reference image is a previous frame (and therefore
  // updated dynamically).
  bool dynamic_ref_;
  // The first element in each pair is the frame id, and the second element is
  // the image data. The frame id is included simple for debugging purposes.
  boost::circular_buffer<std::pair<unsigned long, cv::Mat>> buffer_;
  // Used when reading a static reference image from disk.
  cv::Mat ref_img_;
};

#endif  // STREAMER_PROCESSOR_DIFFERENCE_DETECTOR_H_
