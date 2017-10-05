
#include "utils/output_tracker.h"

#include <sstream>
#include <stdexcept>

#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>

OutputTracker::OutputTracker(const std::string& root_dir, bool organize_by_time,
                             unsigned long frames_per_dir)
    : root_dir_(root_dir),
      organize_by_time_(organize_by_time),
      frames_per_dir_(frames_per_dir),
      frames_in_current_dir_(0),
      current_dirpath_("") {
  if (!boost::filesystem::exists(root_dir_)) {
    std::ostringstream msg;
    msg << "Desired output directory \"" << root_dir_ << "\" does not exist!";
    throw std::runtime_error(msg.str());
  }

  if (!organize_by_time_) {
    ChangeSubdir(0);
  }
}

std::string OutputTracker::GetAndCreateOutputDir(
    boost::posix_time::ptime micros) {
  if (organize_by_time_) {
    // Add subdirectories for date and time.
    std::string date_s =
        boost::gregorian::to_iso_extended_string(micros.date());
    long hours = micros.time_of_day().hours();

    std::ostringstream dirpath;
    dirpath << root_dir_ << "/" << date_s << "/" << hours << "/";

    // Create the output directory, since it might not exist yet.
    boost::filesystem::create_directories(
        boost::filesystem::path{dirpath.str()});
    return dirpath.str();
  } else {
    // If we have filled up the current subdir, then move on to the next one.
    ++frames_in_current_dir_;
    if (frames_in_current_dir_ == frames_per_dir_) {
      ChangeSubdir(current_dir_idx_ + 1);
    }
    return current_dirpath_;
  }
}

void OutputTracker::ChangeSubdir(unsigned long subdir_idx) {
  frames_in_current_dir_ = 0;
  current_dir_idx_ = subdir_idx;

  std::ostringstream current_dirpath;
  current_dirpath << root_dir_ << "/" << current_dir_idx_ << "/";
  current_dirpath_ = current_dirpath.str();
  boost::filesystem::create_directory(
      boost::filesystem::path{current_dirpath_});
}
