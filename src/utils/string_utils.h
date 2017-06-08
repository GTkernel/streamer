//
// Created by Ran Xian (xranthoar@gmail.com) on 9/25/16.
//

#ifndef STREAMER_STRINGUTILS_H
#define STREAMER_STRINGUTILS_H

#include <sstream>

#include <boost/algorithm/string.hpp>

#include "common/common.h"

/**
 * @brief Determine if a string ends with certain suffix.
 *
 * @param str The string to check.
 * @param ending The suffix.
 *
 * @return True if the string ends with ending.
 */
inline bool EndsWith(const string& str, const string& ending) {
  if (ending.size() > str.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), str.rbegin());
}

/**
 * @brief Determine if a string ends with certain prefix.
 *
 * @param str The string to check.
 * @param ending The prefix.
 *
 * @return True if the string starts with <code>start</code>.
 */
inline bool StartsWith(const string& str, const string& start) {
  if (start.size() > str.size()) return false;
  return std::equal(start.begin(), start.end(), str.begin());
}

inline string TrimSpaces(const string& str) {
  size_t first = str.find_first_not_of(' ');
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

inline std::vector<std::string> SplitString(const string& str,
                                            const string& delim) {
  std::vector<string> results;
  boost::split(results, str, boost::is_any_of(delim));
  return results;
}

/**
 * @brief Get protocol name and path from an uri
 * @param uri An uri in the form protocol://path
 * @param protocol The reference to store the protocol.
 * @param path The reference to store the path.
 */
inline void ParseProtocolAndPath(const string& uri, string& protocol,
                                 string& path) {
  std::vector<string> results = SplitString(uri, ":");
  protocol = results[0];
  path = results[1].substr(2);
}

/**
 * @brief Get a numeric value for ip address from a string. 1.2.3.4 will be
 * converted to 0x01020304.
 * @param ip_str The ip address in a string
 * @return The ip address in integer
 */
inline unsigned int GetIPAddrFromString(const string& ip_str) {
  std::vector<string> sp = SplitString(ip_str, ".");
  unsigned int ip_val = 0;
  CHECK(sp.size() == 4) << ip_str << " is not a valid ip address";

  for (int i = 0; i < 4; i++) {
    ip_val += atoi(sp[i].c_str()) << ((3 - i) * 8);
  }

  return ip_val;
}

/**
 * @brief Check if a string contains a substring.
 * @param str The string to be checked.
 * @param substr The substring to be checked.
 * @return Wether the string has the substring or not.
 */
inline bool StringContains(const string& str, const string& substr) {
  return str.find(substr) != string::npos;
}

/**
 * @brief Convert an string to an interger.
 * @param str The string to be converted.
 * @return The converted integer.
 */
inline int StringToInt(const string& str) { return atoi(str.c_str()); }

inline size_t StringToSizet(const string& str) {
  std::istringstream iss(str);
  size_t s;
  iss >> s;
  if (iss.fail()) {
    LOG(FATAL) << "Improperly formed size_t: " << str;
  }
  return s;
}

#endif  // STREAMER_STRINGUTILS_H
