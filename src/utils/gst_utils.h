//
// Created by Ran Xian (xranthoar@gmail.com) on 10/16/16.
//

#ifndef STREAMER_UTILS_GST_UTILS_H_
#define STREAMER_UTILS_GST_UTILS_H_

#include <gst/gst.h>
#include "common/common.h"

/**
 * @brief Check if a gst element exists or not
 *
 * @param element The name of the element
 */
inline bool IsGstElementExists(const string& element) {
  GstElement* ele = gst_element_factory_make(element.c_str(), nullptr);
  bool exists = (ele != nullptr);

  return exists;
}

#endif  // STREAMER_UTILS_GST_UTILS_H_
