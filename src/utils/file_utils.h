
#ifndef STREAMER_UTILS_FILE_UTILS_H_
#define STREAMER_UTILS_FILE_UTILS_H_

#include <fstream>
#include <string>

#include <boost/filesystem.hpp>

#include "utils/time_utils.h"

// Returns true if the specified file exists, or false otherwise.
inline bool FileExists(const std::string& filename) {
  std::ifstream f(filename);
  return f.good();
}

// Returns the directory portion of a filepath.
inline std::string GetDir(const std::string& filepath) {
  size_t last_slash = filepath.find_last_of('/');
  return filepath.substr(0, last_slash + 1);
}

// Creates directories recursively, returning true if successful or false
// otherwise.
inline bool CreateDirs(const std::string& path) {
  if (path == "") return false;
  boost::filesystem::path bpath(path);
  return create_directories(bpath);
}

// Returns the size of the specified file.
inline size_t GetFileSize(const std::string& path) {
  std::ifstream ifile(path);
  ifile.seekg(0, std::ios_base::end);
  size_t size = (size_t)ifile.tellg();
  ifile.close();
  return size;
}

// Creates a directory hierarchy in "base_dir" that is based on the provided
// ptime. The hierarchy includes levels for the day and hour.
inline std::string GetAndCreateDateTimeDir(std::string base_dir,
                                           boost::posix_time::ptime time) {
  std::string dir = GetDateTimeDir(base_dir, time);
  CreateDirs(dir);
  return dir;
}

#endif  // STREAMER_UTILS_FILE_UTILS_H_
