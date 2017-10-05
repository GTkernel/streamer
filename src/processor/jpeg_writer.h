
#ifndef STREAMER_PROCESSOR_JPEG_WRITER_H_
#define STREAMER_PROCESSOR_JPEG_WRITER_H_

#include <memory>
#include <string>

#include "common/types.h"
#include "processor/processor.h"
#include "utils/output_tracker.h"

// This Processor encodes a specified field from each frame as a JPEG file using
// the default JPEG compression settings. Therefore, the specified field must
// represent an image stored as a cv::Mat. The resulting files are named using
// the frame's "capture_time_micros" field and the name of the field that is
// being written.
class JpegWriter : public Processor {
 public:
  // "field" denotes which field to encode, "output_dir" denotes the directory
  // in which the resulting files will be written, and "num_frames_per_dir" is
  // number of frames to put in each subdir in "output_dir".
  JpegWriter(const std::string& field = "original_image",
             const std::string& output_dir = ".", bool organize_by_time = false,
             unsigned long frames_per_dir = 1000);

  // "params" should contain two fields, "field" and "output_dir"
  static std::shared_ptr<JpegWriter> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // The frame field that will be encoded.
  std::string field_;
  // Tracks which directory frames should be written to.
  OutputTracker tracker_;
};

#endif  // STREAMER_PROCESSOR_JPEG_WRITER_H_
