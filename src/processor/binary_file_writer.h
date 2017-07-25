
#ifndef STREAMER_PROCESSOR_BINARY_FILE_WRITER_H_
#define STREAMER_PROCESSOR_BINARY_FILE_WRITER_H_

#include <memory>
#include <string>

#include "common/types.h"
#include "processor/processor.h"

// This Processor writes a specified field, which must be a vector of chars,
// from each frame to disk. The resulting files are named using the frame's
// "frame_id" field.
//
// TODO: Add support for custom filenames.
class BinaryFileWriter : public Processor {
 public:
  // "key" denotes which field to save, and "output_dir" denotes the directory
  // in which the resulting files will be written.
  BinaryFileWriter(std::string key = "original_bytes",
                   std::string output_dir = ".");

  // "params" should contain two keys, "key" and "output_dir".
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
  std::string key_;
  // The destination directory for output files.
  std::string output_dir_;
};

#endif  // STREAMER_PROCESSOR_BINARY_FILE_WRITER_H_