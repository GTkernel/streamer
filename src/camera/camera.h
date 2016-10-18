//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#ifndef TX1DNN_CAMERA_H
#define TX1DNN_CAMERA_H

#include "common/common.h"
#include "processor/processor.h"
#include "stream/stream.h"
#include "video/gst_video_capture.h"

/**
 * @brief This class represents a camera available on the device.
 */
class Camera : public Processor {
 public:
  Camera(){};
  Camera(const string &name, const string &video_uri, int width = -1,
         int height = -1);  // Just a nonsense default value
  string GetName() const;
  string GetVideoURI() const;
  std::shared_ptr<Stream> GetStream() const;
  int GetWidth();
  int GetHeight();

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string name_;
  string video_uri_;
  int width_;
  int height_;
  GstVideoCapture capture_;
  // Camera outout stream
  std::shared_ptr<Stream> stream_;
};

#endif  // TX1DNN_CAMERA_H
