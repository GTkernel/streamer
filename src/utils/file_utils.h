
#ifndef STREAMER_UTILS_FILE_UTILS_H_
#define STREAMER_UTILS_FILE_UTILS_H_

#include <fstream>
#include <string>

#include <boost/filesystem.hpp>

#include "utils/time_utils.h"

/**
 * @brief Check if a file exists or not
 */
inline bool FileExists(const std::string& filename) {
  std::ifstream f(filename);
  return f.good();
}

/**
 * @brief Get the dir part of a filename
 */
inline std::string GetDir(const std::string& filename) {
  size_t last = filename.find_last_of('/');
  return filename.substr(0, last + 1);
}

/**
 * @brief Create directories recursively
 *
 * @return True on success, false otherwise
 */
inline bool CreateDirs(const std::string& path) {
  if (path == "") return false;
  boost::filesystem::path bpath(path);
  return create_directories(bpath);
}

/**
 * @brief Get the size of a file.
 * @param path The path to the file.
 * @return The size of the file in bytes.
 */
inline size_t GetFileSize(const std::string& path) {
  std::ifstream ifile(path);
  ifile.seekg(0, std::ios_base::end);
  size_t size = (size_t)ifile.tellg();
  ifile.close();
  return size;
}

inline std::string GetAndCreateDateTimeDir(std::string base_dir,
                                           boost::posix_time::ptime time) {
  std::string dir = GetDateTimeDir(base_dir, time);
  CreateDirs(dir);
  return dir;
}

#endif  // STREAMER_UTILS_FILE_UTILS_H_
