
#ifndef STREAMER_PROCESSOR_FRAME_WRITER_H_
#define STREAMER_PROCESSOR_FRAME_WRITER_H_

#include <fstream>
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

  std::unordered_set<std::string> fields_;
  std::string output_dir_;
  FileFormat format_;
};

#endif  // STREAMER_PROCESSOR_FRAME_WRITER_H_
