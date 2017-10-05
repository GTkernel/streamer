
#ifndef STREAMER_UTILS_OUTPUT_TRACKER_H_
#define STREAMER_UTILS_OUTPUT_TRACKER_H_

#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>

class OutputTracker {
 public:
  OutputTracker(const std::string& root_dir, bool organize_by_time,
                unsigned long frames_per_dir);
  std::string GetAndCreateOutputDir(boost::posix_time::ptime micros);

 private:
  void ChangeSubdir(unsigned long subdir_idx);

  std::string root_dir_;
  bool organize_by_time_;

  unsigned long frames_per_dir_;
  unsigned long frames_in_current_dir_;
  unsigned long current_dir_idx_;
  std::string current_dirpath_;
};

#endif  // STREAMER_UTILS_OUTPUT_TRACKER_H_
