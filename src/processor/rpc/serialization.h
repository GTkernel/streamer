#ifndef STREAMER_RPC_SERIALIZATION_H
#define STREAMER_RPC_SERIALIZATION_H

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

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

}  // namespace serialization
}  // namespace boost

#endif  // STREAMER_RPC_SERIALIZATION_H
