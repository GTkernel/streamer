
#ifndef STREAMER_UTILS_TIME_UTILS_H_
#define STREAMER_UTILS_TIME_UTILS_H_

#include <ctime>

constexpr unsigned int buf_len = 128;
inline std::string GetCurrentTimeString(const std::string& time_format) {
  std::time_t t = std::time(nullptr);
  char timestr[buf_len];

  std::strftime(timestr, sizeof(timestr), time_format.c_str(),
                std::localtime(&t));

  return std::string(timestr);
}

#endif  // STREAMER_UTILS_TIME_UTILS_H_
