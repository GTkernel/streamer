/**
* Multi-target tracking using Struck
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/lexical_cast.hpp>
#include "common/context.h"
#include "struck_tracker.h"

static const string STRUCK_CONF_FILENAME = "struck_config.txt";

StruckTracker::StruckTracker(float calibration_duration)
    : Processor({"input"}, {"output"}),
      conf_(Context::GetContext().GetConfigFile(STRUCK_CONF_FILENAME)),
      calibration_duration_(calibration_duration) {}

bool StruckTracker::Init() {
  /*
  string struck_config_path =
      Context::GetContext().GetConfigFile(STRUCK_CONF_FILENAME);
  conf_.quietMode = true;
  conf_.debugMode = false;
  conf_.frameWidth = shape_.width;
  conf_.frameHeight = shape_.height;
  conf_.seed = 0;
  conf_.searchRadius = 30;
  conf_.svmC = 100.0;
  conf_.svmBudgetSize = 100;
  struck::Config::FeatureKernelPair fkp;
  fkp.feature = struck::Config::kFeatureTypeHaar;
  fkp.kernel = struck::Config::kKernelTypeGaussian;
  fkp.params.push_back(0.2);
  conf_.features.push_back(fkp);

  struck::Config conf(struck_config_path);
  */
	std::cout << conf_ << std::endl;

  LOG(INFO) << "StruckTracker initialized";
  return true;
}

bool StruckTracker::OnStop() {
  tracker_list_.clear();
  return true;
}

void StruckTracker::Process() {
  Timer timer;
  timer.Start();
  
  auto md_frame = GetFrame<MetadataFrame>("input");
  cv::Mat image = md_frame->GetOriginalImage();
  if (image.channels() == 3) {
    cv::cvtColor(image, gray_image_, cv::COLOR_BGR2GRAY);
  } else {
    gray_image_ = image;
  }

  std::vector<Rect> tracked_bboxes;
  std::vector<string> tracked_tags;
  std::vector<string> tracked_uuids;
  if (md_frame->GetBitset().test(MetadataFrame::Bit_bboxes)) {
    std::vector<Rect> bboxes = md_frame->GetBboxes();
    LOG(INFO) << "Got new MetadataFrame, bboxes size is " << bboxes.size()
              << ", current tracker size is " << tracker_list_.size();
    std::vector<Rect> untracked_bboxes = bboxes;
    std::vector<string> untracked_tags = md_frame->GetTags();
    CHECK(untracked_bboxes.size() == untracked_tags.size());
    for (auto it = tracker_list_.begin(); it != tracker_list_.end(); ) {
      (*it)->Track(gray_image_);
      struck::IntRect r((*it)->GetBB());
      cv::Rect rt(r.XMin(), r.YMin(), r.Width(), r.Height());
      float best_percent = 0.f;
      //for (auto u_it = untracked_bboxes.begin(); u_it != untracked_bboxes.end(); ++u_it) {
      for (size_t i = 0; i < untracked_bboxes.size(); ++i) {
        cv::Rect ru(untracked_bboxes[i].px, untracked_bboxes[i].py,
            untracked_bboxes[i].width, untracked_bboxes[i].height);
        cv::Rect intersects = rt&ru;
        float percent = (float)intersects.area() / (float)ru.area();
        if (percent >= 0.7) {
          untracked_bboxes.erase(untracked_bboxes.begin()+i);
          untracked_tags.erase(untracked_tags.begin()+i);
          best_percent = percent;
          break;
        }
      }
      if (best_percent >= 0.7) {
        tracked_bboxes.push_back(Rect(r.XMin(), r.YMin(), r.Width(), r.Height()));
        tracked_tags.push_back((*it)->GetTag());
        tracked_uuids.push_back((*it)->GetUuid());
        it++;
      } else {
        LOG(INFO) << "Remove tracker, best_percent is " << best_percent;
        tracker_list_.erase(it++);
      }
    }

    CHECK(untracked_bboxes.size() == untracked_tags.size());
    //for (const auto& m: untracked_bboxes) {
    for (size_t i = 0; i < untracked_bboxes.size(); ++i) {
      LOG(INFO) << "Create new tracker";
      int x = untracked_bboxes[i].px;
      int y = untracked_bboxes[i].py;
      int w = untracked_bboxes[i].width;
      int h = untracked_bboxes[i].height;
      CHECK((x>=0) && (y>=0) && (x+w<=gray_image_.cols) && (y+h<=gray_image_.rows));
      struck::FloatRect initBB = struck::IntRect(x, y, w, h);
      boost::uuids::uuid uuid = boost::uuids::random_generator()();
      std::string uuid_str = boost::lexical_cast<std::string>(uuid);
      std::shared_ptr<Tracker1> new_tracker(new Tracker1(
          conf_, uuid_str, untracked_tags[i]));
      new_tracker->Initialise(gray_image_, initBB);
      CHECK(new_tracker->IsInitialised());
      //printf("%s, %d\n", __FUNCTION__, __LINE__);
      new_tracker->Track(gray_image_);
      //printf("%s, %d\n", __FUNCTION__, __LINE__);
      struck::IntRect r(new_tracker->GetBB());
      tracked_bboxes.push_back(Rect(r.XMin(), r.YMin(), r.Width(), r.Height()));
      tracked_tags.push_back(untracked_tags[i]);
      tracked_uuids.push_back(uuid_str);
      tracker_list_.push_back(new_tracker);
    }
    last_calibration_time_ = std::chrono::system_clock::now();
  } else {
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = now-last_calibration_time_;
    if (diff.count() < calibration_duration_) {
      for (auto it = tracker_list_.begin(); it != tracker_list_.end(); ++it) {
        (*it)->Track(gray_image_);
        struck::IntRect r((*it)->GetBB());
        tracked_bboxes.push_back(Rect(r.XMin(), r.YMin(), r.Width(), r.Height()));
        tracked_tags.push_back((*it)->GetTag());
        tracked_uuids.push_back((*it)->GetUuid());
      }
    } else {
      LOG(INFO) << "Time " << calibration_duration_ << " is up, need calibration ......";
      return;
    }
  }

  md_frame->SetBboxes(tracked_bboxes);
  md_frame->SetTags(tracked_tags);
  md_frame->SetUuids(tracked_uuids);
  PushFrame("output", md_frame);
  LOG(INFO) << "StruckTracker took " << timer.ElapsedMSec() << " ms";
}

ProcessorType StruckTracker::GetType() {
  return PROCESSOR_TYPE_STRUCK_TRACKER;
}
