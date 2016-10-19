//
// Created by Ran Xian (xranthoar@gmail.com) on 10/16/16.
//

#ifndef TX1DNN_GST_UTILS_H
#define TX1DNN_GST_UTILS_H
#include <gst/gst.h>
#include "common/common.h"

/**
 * @brief Check if a gst element exists or not
 *
 * @param element The name of the element
 */
inline bool IsGstElementExists(const string &element) {
  GstElement *ele = gst_element_factory_make(element.c_str(), nullptr);
  bool exists = (ele != nullptr);

  return exists;
}

#endif  // TX1DNN_FILE_UTILS_H