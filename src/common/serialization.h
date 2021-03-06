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

#ifndef STREAMER_COMMON_SERIALIZATION_H_
#define STREAMER_COMMON_SERIALIZATION_H_

#include <boost/date_time/posix_time/time_serialize.hpp>
#include <boost/serialization/array.hpp>
#include <opencv2/opencv.hpp>
#ifdef USE_TENSORFLOW
#include "tensorflow/core/framework/tensor.h"
#endif  // USE_TENSORFLOW

namespace boost {
namespace serialization {

/** Serialization support for cv::Mat */
// http://stackoverflow.com/a/21444792/1072039
template <class Archive>
void serialize(Archive& ar, cv::Mat& mat, const unsigned int) {
  int cols, rows, channels, type;
  // Weird mode is when opencv decides that there are -1 rows and -1 columns.
  // In most cases, there are more than -1 rows and -1 columns, but
  // for some reason opencv wants to keep it a secret.
  bool weird_mode = false;

  if (Archive::is_saving::value) {
    rows = mat.rows;
    cols = mat.cols;
    channels = mat.channels();
    type = mat.type();
    if (cols < 0 || rows < 0) {
      rows = mat.size[0];
      cols = mat.size[1];
      channels = mat.size[2];
      weird_mode = true;
    }
  }

  ar& cols& rows& type& channels;

  if (Archive::is_loading::value) {
    // If opencv doesn't cooperate then force it to work in a different way
    if (weird_mode) {
      // TODO: make sure type is not a special type that includes channels
      mat.create({rows, cols, channels}, type);
    } else {
      mat.create(rows, cols, type);
    }
  }
  bool continuous = mat.isContinuous();

  // Too lazy to implement both branches for weird mode
  // First branch is just a performance optimization anyways
  if (continuous && !weird_mode) {
    int data_size = rows * cols * mat.elemSize();
    ar& boost::serialization::make_array(mat.ptr(), data_size);
  } else {
    int row_size = cols * mat.elemSize();
    if (weird_mode) {
      row_size *= channels;
    }
    for (int i = 0; i < rows; i++) {
      ar& boost::serialization::make_array(mat.ptr(i), row_size);
    }
  }
}

#ifdef USE_TENSORFLOW
/** Serialization support for Tensor */
template <class Archive>
void serialize(Archive& ar, tensorflow::Tensor& tensor, const unsigned int) {
  // assume tensor is always continuous
  int vec_size, height, width, channels;
  tensorflow::DataType type;

  if (Archive::is_saving::value) {
    type = tensor.dtype();
    vec_size = tensor.dim_size(0);
    height = tensor.dim_size(1);
    width = tensor.dim_size(2);
    channels = tensor.dim_size(3);
  }
  ar& type& vec_size& height& width& channels;

  const unsigned int data_size = vec_size * height * width * channels;
  if (Archive::is_loading::value) {
    tensorflow::Tensor reshape_tensor(
        type, tensorflow::TensorShape(
                  {static_cast<long long>(vec_size), height, width, channels}));
    bool success = reshape_tensor.CopyFrom(
        tensor, tensorflow::TensorShape({static_cast<long long>(vec_size),
                                         height, width, channels}));
    if (!success) {
      std::ostringstream msg;
      msg << "Problem copying Tensor, shapes do not agree! Source: [ ";
      for (auto d : tensor.shape().dim_sizes()) {
        msg << d << " ";
      }
      msg << "],  Destination: ";
      for (auto d : reshape_tensor.shape().dim_sizes()) {
        msg << d << " ";
      }
      msg << "]";
  //    throw std::runtime_error(msg.str());
    }
    tensor = reshape_tensor;
  }

  ar& boost::serialization::make_array(tensor.flat<float>().data(), data_size);
}
#endif  // USE_TENSORFLOW

}  // namespace serialization
}  // namespace boost

#endif  // STREAMER_COMMON_SERIALIZATION_H_
