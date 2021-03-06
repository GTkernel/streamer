// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STREAMER_UTILS_GST_UTILS_H_
#define STREAMER_UTILS_GST_UTILS_H_

#include <gst/gst.h>

/**
 * @brief Check if a gst element exists or not
 *
 * @param element The name of the element
 */
inline bool IsGstElementExists(const std::string& element) {
  GstElement* ele = gst_element_factory_make(element.c_str(), nullptr);
  bool exists = (ele != nullptr);

  return exists;
}

#endif  // STREAMER_UTILS_GST_UTILS_H_
