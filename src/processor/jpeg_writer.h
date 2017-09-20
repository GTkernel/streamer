
#ifndef STREAMER_PROCESSOR_JPEG_WRITER_H_
#define STREAMER_PROCESSOR_JPEG_WRITER_H_

#include <memory>
#include <string>

#include "common/types.h"
#include "processor/processor.h"

// This Processor encodes a specified field from each frame as a JPEG file using
// the default JPEG compression settings. Therefore, the specified field must
// represent an image stored as a cv::Mat. The resulting files are named using
// the frame's "frame_id" field.
//
// TODO: Add support for custom filenames.
class JpegWriter : public Processor {
 public:
  // "key" denotes which field to encode, "output_dir" denotes the directory
  // in which the resulting files will be written, and "num_frames_per_dir" is
  // number of frames to put in each subdir in "output_dir".
  JpegWriter(std::string key = "original_image", std::string output_dir = ".",
             unsigned long num_frames_per_dir = 1000);

  // "params" should contain two keys, "key" and "output_dir"
  static std::shared_ptr<JpegWriter> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // The frame field that will be encoded.
  std::string key_;
  // The destination directory for output files.
  std::string output_dir_;
  // The number of frames to save in each output subdir.
  unsigned long num_frames_per_dir_;
};

#endif  // STREAMER_PROCESSOR_JPEG_WRITER_H_
