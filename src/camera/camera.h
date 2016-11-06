//
// Created by Ran Xian (xranthoar@gmail.com) on 9/23/16.
//

#ifndef STREAMER_CAMERA_H
#define STREAMER_CAMERA_H

#include "common/common.h"
#include "processor/processor.h"
#include "stream/stream.h"

/**
 * @brief This class represents a camera available on the device.
 */
class Camera : public Processor {
 public:
  Camera(const string &name, const string &video_uri, int width = -1,
         int height = -1, int nsink = 1);  // Just a nonsense default value
  string GetName() const;
  string GetVideoURI() const;
  std::shared_ptr<Stream> GetStream() const;
  int GetWidth();
  int GetHeight();
  virtual CameraType GetType() const = 0;
  virtual ProcessorType GetType() override;

 protected:
  virtual bool Init() = 0;
  virtual bool OnStop() = 0;
  virtual void Process() = 0;

 protected:
  string name_;
  string video_uri_;
  int width_;
  int height_;
  // Camera output stream
  std::shared_ptr<Stream> stream_;
};

#endif  // STREAMER_CAMERA_H
