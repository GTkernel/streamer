/**
 * Multi-target tracking using dlib.
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#ifndef STREAMER_PROCESSOR_DLIB_TRACKER_H_
#define STREAMER_PROCESSOR_DLIB_TRACKER_H_

#include <dlib/dlib/image_processing.h>
#include <dlib/dlib/opencv.h>

class DlibTracker : public BaseTracker {
 public:
  DlibTracker(const std::string& uuid, const std::string& tag)
      : BaseTracker(uuid, tag) {
    impl_.reset(new dlib::correlation_tracker());
  }
  virtual ~DlibTracker() {}
  virtual void Initialise(const cv::Mat& gray_image, cv::Rect bb) {
    dlib::array2d<unsigned char> dlibImageGray;
    dlib::assign_image(dlibImageGray,
                       dlib::cv_image<unsigned char>(gray_image));
    dlib::rectangle initBB(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);
    impl_->start_track(dlibImageGray, initBB);
  }
  virtual bool IsInitialised() { return true; }
  virtual void Track(const cv::Mat& gray_image) {
    dlib::array2d<unsigned char> dlibImageGray;
    dlib::assign_image(dlibImageGray,
                       dlib::cv_image<unsigned char>(gray_image));
    impl_->update(dlibImageGray);
  }
  virtual cv::Rect GetBB() {
    auto r = impl_->get_position();
    return cv::Rect(r.left(), r.top(), r.right() - r.left(),
                    r.bottom() - r.top());
  }
  virtual std::vector<double> GetBBFeature() {
    return std::vector<double>(128, 0.f);
  }

 private:
  std::unique_ptr<dlib::correlation_tracker> impl_;
};

#endif  // STREAMER_PROCESSOR_DLIB_TRACKER_H_