//
// Created by Ran Xian (xranthoar@gmail.com) on 11/13/16.
//

#ifndef STREAMER_FILE_WRITER_H
#define STREAMER_FILE_WRITER_H

#include <stdlib.h>
#include <fstream>

#include "processor.h"

/**
 * @brief A file writer that writes raw bytes to a file.
 */
class FileWriter : public Processor {
 public:
  /**
   * @brief FileWriter constructor
   * @param filename The name of the file.
   */
  FileWriter(const string &filename);

  virtual ProcessorType GetType() override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string filename_;
  std::ofstream file_;
};

#endif  // STREAMER_FILE_WRITER_H
