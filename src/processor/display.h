
#ifndef STREAMER_PROCESSOR_DISPLAY_H_
#define STREAMER_PROCESSOR_DISPLAY_H_

#include <memory>
#include <string>

#include "common/types.h"
#include "processor/processor.h"

// Displays frames in an OpenCV window at a specified size ratio and rotation
// angle. After being displayed, frames are passed unchanged to the next
// Processor.
class Display : public Processor {
 public:
  // "key" is the frame key to display, which must correspond to a cv::Mat
  // object. "window_name" is the name to give the display window, which must be
  // unique.
  Display(const std::string& key, unsigned int angle, float size_ratio,
          const std::string& window_name);
  static std::shared_ptr<Display> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::string key_;
  unsigned int angle_;
  float size_ratio_;
  std::string window_name_;
};

#endif  // STREAMER_PROCESSOR_DISPLAY_H_
