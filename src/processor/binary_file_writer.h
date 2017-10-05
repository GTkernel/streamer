
#ifndef STREAMER_PROCESSOR_BINARY_FILE_WRITER_H_
#define STREAMER_PROCESSOR_BINARY_FILE_WRITER_H_

#include <memory>
#include <string>

#include "common/types.h"
#include "processor/processor.h"
#include "utils/output_tracker.h"

// This Processor writes a specified field, which must be a vector of chars,
// from each frame to disk. The resulting files are named using the frame's
// "capture_time_micros" field and the name of the field that is being written.
//
// TODO: Add support for custom filenames.
class BinaryFileWriter : public Processor {
 public:
  // "field" denotes which field to save, and "output_dir" denotes the directory
  // in which the resulting files will be written.
  BinaryFileWriter(const std::string& field = "original_bytes",
                   const std::string& output_dir = ".",
                   bool organize_by_time = false,
                   unsigned long frames_per_dir = 1000);

  // "params" should contain two fields, "field" and "output_dir".
  static std::shared_ptr<BinaryFileWriter> Create(
      const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // The frame field that will be saved.
  std::string field_;
  // Tracks which directory frames should be written to.
  OutputTracker tracker_;
};

#endif  // STREAMER_PROCESSOR_BINARY_FILE_WRITER_H_
