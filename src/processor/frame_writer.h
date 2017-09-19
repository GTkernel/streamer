
#ifndef STREAMER_PROCESSOR_FRAME_WRITER_H_
#define STREAMER_PROCESSOR_FRAME_WRITER_H_

#include <memory>
#include <string>
#include <unordered_set>

#include "common/types.h"
#include "processor/processor.h"

// The FrameWriter writes Frames to disk in either text or binary format. The
// user can specify which frame fields to save (the default is all fields).
class FrameWriter : public Processor {
 public:
  enum FileFormat { BINARY, JSON, TEXT };

  // "fields" is a set of frame fields to save. If "fields" is an empty set,
  // then all fields will be saved.
  FrameWriter(const std::unordered_set<std::string> fields = {},
              const std::string& output_dir = ".",
              unsigned int frames_per_dir = 1000,
              const FileFormat format = TEXT);

  // "params" should contain three keys, "fields" (which should be a set of
  // field names), "output_dir", and "format" (which should be the textual
  // representation of the FileFormat value).
  static std::shared_ptr<FrameWriter> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::string GetExtension();
  void SetSubdir(unsigned int subdir);

  // The frame fields to save.
  std::unordered_set<std::string> fields_;
  // The root output directory filepath.
  std::string output_dir_;
  // The filepath of the current output subdirectory.
  std::string output_subdir_;
  // The maximum number of frames to store in each output subdirectory.
  unsigned int frames_per_dir_;
  // The number of frames that have already been stored in the current output
  // subdirectory.
  unsigned int frames_in_current_dir_;
  // The numeric name of the current output subdirectory.
  unsigned int current_dir_;
  // The file format in which to save frames.
  FileFormat format_;
};

#endif  // STREAMER_PROCESSOR_FRAME_WRITER_H_
