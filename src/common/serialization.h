
#ifndef STREAMER_COMMON_SERIALIZATION_H_
#define STREAMER_COMMON_SERIALIZATION_H_

#include <boost/date_time/posix_time/time_serialize.hpp>
#include <boost/serialization/array.hpp>
#include <opencv2/opencv.hpp>

#include "tensorflow/core/framework/tensor.h"

namespace boost {
namespace serialization {

/** Serialization support for cv::Mat */
// http://stackoverflow.com/a/21444792/1072039
template <class Archive>
void serialize(Archive& ar, cv::Mat& mat, const unsigned int) {
  int cols, rows, type;
  bool continuous;

  if (Archive::is_saving::value) {
    cols = mat.cols;
    rows = mat.rows;
    type = mat.type();
    continuous = mat.isContinuous();
  }

  ar& cols& rows& type& continuous;

  if (Archive::is_loading::value) mat.create(rows, cols, type);

  if (continuous) {
    const unsigned int data_size = rows * cols * mat.elemSize();
    ar& boost::serialization::make_array(mat.ptr(), data_size);
  } else {
    const unsigned int row_size = cols * mat.elemSize();
    for (int i = 0; i < rows; i++) {
      ar& boost::serialization::make_array(mat.ptr(i), row_size);
    }
  }
}

/** TODO Serialization support for Tensor */
template <class Archive>
void serialize(Archive& ar, tensorflow::Tensor& tensor, const unsigned int) {
	//assume tensor is always continuous
	int vec_size, height, width, channels;
	tensorflow::DataType type;

	if (Archive::is_saving::value){
		type = tensor.dtype();
		vec_size = tensor.dim_size(0);
		height = tensor.dim_size(1);
		width = tensor.dim_size(2);
		channels = tensor.dim_size(3);
	}
	ar& type& vec_size& height& width& channels;

	const unsigned int data_size = vec_size * height * width * channels;

	if (Archive::is_loading::value) {
		tensorflow::Tensor reshape_tensor(type,\
										  tensorflow::TensorShape({static_cast<long long>(vec_size),\
										                           height, width, channels}\
                                          ));
		reshape_tensor.CopyFrom(tensor, tensorflow::TensorShape({static_cast<long long>(vec_size),height, width, channels}));
		tensor = reshape_tensor;
	}

	ar& boost::serialization::make_array(tensor.flat<float>().data(), data_size);
}


}  // namespace serialization
}  // namespace boost

#endif  // STREAMER_COMMON_SERIALIZATION_H_
