//
// Created by Ran Xian (xranthoar@gmail.com) on 11/2/16.
//

#ifndef STREAMER_FILE_WRITER_H
#define STREAMER_FILE_WRITER_H

#include "streamer.h"

#include <stdlib.h>
#include <fstream>

/**
 * @brief A file writer that writes raw bytes to a file.
 */
class FileWriter : public Processor {
 public:
  /**
   * @brief FileWriter constructor
   * @param Input stream that feeds bytes.
   * @param filename_base The basename of a file, filename will be named as
   * ${basename}-${start_frame}-${end_frame}.
   * @param frames_per_file Number of frames per file.
   */
  FileWriter(const string &filename_base, size_t frames_per_file);

  size_t GetFramesWritten() { return frames_written_; }
  string GetCurrentFilename() { return current_filename_; }
  virtual ProcessorType GetType() override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string filename_base_;
  string directory_name_;
  string current_filename_;
  size_t frames_written_;
  size_t frames_per_file_;

  std::ofstream current_file_;
};

#endif  // STREAMER_FILE_WRITER_H
