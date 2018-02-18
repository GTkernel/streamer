
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

  // Signals the DiffDetector to begin logging the "frame_id" field of each
  // approved frame to a file. The file will be written to the directory
  // "output_dir" and will be named
  // "diff_detector_<global/blocked>_<static/dynamic>_<threshold>.log".
  void EnableLog(std::string output_dir);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  static double GlobalMse(cv::Mat img, cv::Mat ref_img);
  static double BlockedMse(cv::Mat img, cv::Mat ref_img, cv::Mat weights,
                           int block_size);
  // Load the weights used in the blocked difference detector.
  static cv::Mat LoadWeights(const std::string& weights_path);
  // Load the static reference image.
  static cv::Mat ReadRefImg(const std::string& ref_path);

  double threshold_;

  bool blocked_;
  int block_size_;
  cv::Mat weights_;

  // Indicates whether the reference image is a previous frame (and therefore
  // updated dynamically).
  bool dynamic_ref_;
  unsigned long t_diff_frames_;
  // The first element in each pair is the frame id, and the second element is
  // the image data. The frame id is included simply for debugging purposes.
  boost::circular_buffer<std::pair<unsigned long, cv::Mat>> buffer_;
  // Used when reading a static reference image from disk.
  cv::Mat ref_img_;

  // If this stream is open, then the frame selection results will be written to
  // it.
  std::ofstream log_;
};

#endif  // STREAMER_PROCESSOR_DIFFERENCE_DETECTOR_H_
