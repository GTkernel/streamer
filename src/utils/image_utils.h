#ifndef STREAMER_UTILS_IMAGE_UTILS_H_
#define STREAMER_UTILS_IMAGE_UTILS_H_

/**
 * @brief Rotate an OpenCV image matrix
 * @param m The OpenCV matrix
 * @param angle The angle to rotate; must be 0, 90, 180, or 270
 */
inline void RotateImage(cv::Mat& m, const unsigned int angle) {
  CHECK(angle == 0 || angle == 90 || angle == 180 || angle == 270)
      << "; angle was " << angle;

  if (angle == 90) {
    cv::transpose(m, m);
    cv::flip(m, m, 1);
  } else if (angle == 180) {
    cv::flip(m, m, -1);
  } else if (angle == 270) {
    cv::transpose(m, m);
    cv::flip(m, m, 0);
  }
}

#endif  // STREAMER_UTILS_IMAGE_UTILS_H_
