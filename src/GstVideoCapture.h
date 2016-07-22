//
// Created by Ran Xian on 7/22/16.
//

#ifndef TX1_DNN_GSTVIDEOCAPTURE_H
#define TX1_DNN_GSTVIDEOCAPTURE_H

#include "common.h"
#include <opencv2/opencv.hpp>
/**
 * Video capture for reading frames from GStreamer. Return frames in OpenCV BGR Mat. Internally the video capture is
 * using GStreamer-1.0's appsink to `intercept' frame buffers. When the \b GetFrame() method is called, The video
 * capture will issue a pre_preroll signal to the GStreamer pipeline.
 */
class GstVideoCapture {
public:
    GstVideoCapture() = delete;
    GstVideoCapture(std::string rstp_uri);
    cv::Mat GetFrame();
};


#endif //TX1_DNN_GSTVIDEOCAPTURE_H
