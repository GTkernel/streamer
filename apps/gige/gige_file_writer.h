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
class GigeFileWriter : public Processor {
 public:
  /**
   * @brief FileWriter constructor
   * @param directory Directroy to store the file.
   * @param filename_base The basename of a file, file will be named as
   * {frame_count / frames_per_file}.dat
   * @param frames_per_file Number of frames per file.
   */
  GigeFileWriter(const string& directory, size_t frames_per_file);

  size_t GetFramesWritten() { return frames_written_; }
  string GetCurrentFilename() { return current_filename_; }
  string GetCurrentDirectory() { return directory_; }
  void SetDirectory(const string& directory) { directory_ = directory; }

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string directory_;
  string current_filename_;
  size_t frames_written_;
  size_t frames_per_file_;

  std::ofstream current_file_;
};

#endif  // STREAMER_FILE_WRITER_H
