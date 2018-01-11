
#ifndef STREAMER_UTILS_TIME_UTILS_H_
#define STREAMER_UTILS_TIME_UTILS_H_

#include <sstream>
#include <string>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

// Returns a string encoding of the provided ptime.
inline std::string GetDateTimeString(boost::posix_time::ptime time) {
  return boost::posix_time::to_iso_extended_string(time);
}

// Returns a string encoding of the current time.
inline std::string GetCurrentDateTimeString() {
  return GetDateTimeString(boost::posix_time::microsec_clock::local_time());
}

// Returns a string specifying the full directory path, starting with
// "base_dir", that ends with a hierarchy based on the provided ptime. The
// hierarchy includes levels for the day and hour.
inline std::string GetDateTimeDir(std::string base_dir,
                                  boost::posix_time::ptime time) {
  std::string date = boost::gregorian::to_iso_extended_string(time.date());
  long hours = time.time_of_day().hours();

  std::ostringstream dirpath;
  dirpath << base_dir << "/" << date << "/" << hours << "/";
  return dirpath.str();
}

#endif  // STREAMER_UTILS_TIME_UTILS_H_
