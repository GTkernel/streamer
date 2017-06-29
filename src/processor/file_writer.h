//
// Created by Ran Xian (xranthoar@gmail.com) on 11/13/16.
//

#ifndef STREAMER_PROCESSOR_FILE_WRITER_H_
#define STREAMER_PROCESSOR_FILE_WRITER_H_

#include <stdlib.h>
#include <fstream>

#include "common/types.h"
#include "processor/processor.h"

/**
 * @brief A file writer that writes raw bytes to a file.
 */
class FileWriter : public Processor {
 public:
  enum file_format { TEXT, BINARY };

  /**
   * @brief FileWriter constructor
   * @param filename The name of the file.
   */
  FileWriter(const string& filename, const file_format format = TEXT);

  static std::shared_ptr<FileWriter> Create(const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string filename_;
  std::ofstream file_;
  file_format format_;
};

#endif  // STREAMER_PROCESSOR_FILE_WRITER_H_
