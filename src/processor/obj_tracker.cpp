/**
 * Multi-target tracking using Struck etc.
 *
 * @author Tony Chen <xiaolongx.chen@intel.com>
 * @author Shao-Wen Yang <shao-wen.yang@intel.com>
 */

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.
#include "common/context.h"
#include "struck/src/Config.h"
#include "struck/src/Tracker.h"
#ifdef USE_DLIB
#include <dlib/dlib/image_processing.h>
#include <dlib/dlib/opencv.h>
#endif
#include "obj_tracker.h"

static const string STRUCK_CONF_FILENAME = "struck_config.txt";

class StruckTracker : public BaseTracker {
 public:
  StruckTracker(const std::string& uuid, const std::string& tag)
      : BaseTracker(uuid, tag),
        conf_(Context::GetContext().GetConfigFile(STRUCK_CONF_FILENAME)) {
    impl_.reset(new struck::Tracker(conf_));
  }
  virtual ~StruckTracker() {}
  virtual void Initialise(const cv::Mat& gray_image, cv::Rect bb) {
    struck::FloatRect initBB = struck::IntRect(bb.x, bb.y, bb.width, bb.height);
    impl_->Initialise(gray_image, initBB);
  }
  virtual bool IsInitialised() { return impl_->IsInitialised(); }
  virtual void Track(const cv::Mat& gray_image) { impl_->Track(gray_image); }
  virtual cv::Rect GetBB() {
    struck::IntRect r(impl_->GetBB());
    return cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height());
  }
  virtual std::vector<double> GetBBFeature() { return impl_->GetBBFeature(); }

 private:
  std::unique_ptr<struck::Tracker> impl_;
  struck::Config conf_;
};

#ifdef USE_DLIB
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
#endif

ObjTracker::ObjTracker(const std::string& type, float calibration_duration)
    : Processor(PROCESSOR_TYPE_OBJ_TRACKER, {"input"}, {"output"}),
      type_(type),
      calibration_duration_(calibration_duration) {}

bool ObjTracker::Init() {
  LOG(INFO) << "ObjTracker initialized";
  return true;
}

bool ObjTracker::OnStop() {
  tracker_list_.clear();
  return true;
}

void ObjTracker::Process() {
  Timer timer;
  timer.Start();

  auto frame = GetFrame("input");
  auto image = frame->GetValue<cv::Mat>("original_image");
  if (image.channels() == 3) {
    cv::cvtColor(image, gray_image_, cv::COLOR_BGR2GRAY);
  } else {
    gray_image_ = image;
  }

  std::vector<Rect> tracked_bboxes;
  std::vector<string> tracked_tags;
  std::vector<string> tracked_uuids;
  std::vector<std::vector<double>> struck_features;
  if (frame->Count("bounding_boxes") > 0) {
    auto bboxes = frame->GetValue<std::vector<Rect>>("bounding_boxes");
    LOG(INFO) << "Got new MetadataFrame, bboxes size is " << bboxes.size()
              << ", current tracker size is " << tracker_list_.size();
    std::vector<Rect> untracked_bboxes = bboxes;
    auto untracked_tags = frame->GetValue<std::vector<std::string>>("tags");
    CHECK(untracked_bboxes.size() == untracked_tags.size());
    for (auto it = tracker_list_.begin(); it != tracker_list_.end();) {
      (*it)->Track(gray_image_);
      cv::Rect rt((*it)->GetBB());
      float best_percent = 0.f;
      // for (auto u_it = untracked_bboxes.begin(); u_it !=
      // untracked_bboxes.end(); ++u_it) {
      for (size_t i = 0; i < untracked_bboxes.size(); ++i) {
        cv::Rect ru(untracked_bboxes[i].px, untracked_bboxes[i].py,
                    untracked_bboxes[i].width, untracked_bboxes[i].height);
        cv::Rect intersects = rt & ru;
        float percent = (float)intersects.area() / (float)ru.area();
        if (percent >= 0.7) {
          untracked_bboxes.erase(untracked_bboxes.begin() + i);
          untracked_tags.erase(untracked_tags.begin() + i);
          best_percent = percent;
          break;
        }
      }
      if (best_percent >= 0.7) {
        tracked_bboxes.push_back(Rect(rt.x, rt.y, rt.width, rt.height));
        tracked_tags.push_back((*it)->GetTag());
        tracked_uuids.push_back((*it)->GetUuid());
        struck_features.push_back((*it)->GetBBFeature());
        it++;
      } else {
        LOG(INFO) << "Remove tracker, best_percent is " << best_percent;
        tracker_list_.erase(it++);
      }
    }

    CHECK(untracked_bboxes.size() == untracked_tags.size());
    // for (const auto& m: untracked_bboxes) {
    for (size_t i = 0; i < untracked_bboxes.size(); ++i) {
      LOG(INFO) << "Create new tracker";
      int x = untracked_bboxes[i].px;
      int y = untracked_bboxes[i].py;
      int w = untracked_bboxes[i].width;
      int h = untracked_bboxes[i].height;
      CHECK((x >= 0) && (y >= 0) && (x + w <= gray_image_.cols) &&
            (y + h <= gray_image_.rows));
      cv::Rect bb(x, y, w, h);
      boost::uuids::uuid uuid = boost::uuids::random_generator()();
      std::string uuid_str = boost::lexical_cast<std::string>(uuid);
      std::shared_ptr<BaseTracker> new_tracker;
      if (type_ == "struck") {
        new_tracker.reset(new StruckTracker(uuid_str, untracked_tags[i]));
      } else if (type_ == "dlib") {
#ifdef USE_DLIB
        new_tracker.reset(new DlibTracker(uuid_str, untracked_tags[i]));
#else
        LOG(FATAL) << "Tracker type " << type_
                   << " not supported, please compile with -DUSE_DLIB=ON";
#endif
      } else {
        LOG(FATAL) << "Tracker type " << type_ << " not supported.";
      }
      new_tracker->Initialise(gray_image_, bb);
      CHECK(new_tracker->IsInitialised());
      // printf("%s, %d\n", __FUNCTION__, __LINE__);
      new_tracker->Track(gray_image_);
      // printf("%s, %d\n", __FUNCTION__, __LINE__);
      cv::Rect rt(new_tracker->GetBB());
      tracked_bboxes.push_back(Rect(rt.x, rt.y, rt.width, rt.height));
      tracked_tags.push_back(untracked_tags[i]);
      tracked_uuids.push_back(uuid_str);
      struck_features.push_back(new_tracker->GetBBFeature());
      tracker_list_.push_back(new_tracker);
    }
    last_calibration_time_ = std::chrono::system_clock::now();
  } else {
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = now - last_calibration_time_;
    if (diff.count() < calibration_duration_) {
      for (auto it = tracker_list_.begin(); it != tracker_list_.end(); ++it) {
        (*it)->Track(gray_image_);
        cv::Rect rt((*it)->GetBB());
        tracked_bboxes.push_back(Rect(rt.x, rt.y, rt.width, rt.height));
        tracked_tags.push_back((*it)->GetTag());
        tracked_uuids.push_back((*it)->GetUuid());
        struck_features.push_back((*it)->GetBBFeature());
      }
    } else {
      LOG(INFO) << "Time " << calibration_duration_
                << " is up, need calibration ......";
      return;
    }
  }

  frame->SetValue("bounding_boxes", tracked_bboxes);
  frame->SetValue("tags", tracked_tags);
  frame->SetValue("uuids", tracked_uuids);
  frame->SetValue("struck_features", struck_features);
  PushFrame("output", std::move(frame));
  LOG(INFO) << "ObjTracker took " << timer.ElapsedMSec() << " ms";
}
