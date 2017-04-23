#include "common/context.h"
#include "struck_tracker.h"

static const string STRUCK_CONF_FILENAME = "struck_config.txt";

StruckTracker::StruckTracker()
    : Processor({"input"}, {"output"}),
      conf_(Context::GetContext().GetConfigFile(STRUCK_CONF_FILENAME)) {}

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
  //Timer timer;
  //timer.Start();
  
  auto md_frame = GetFrame<MetadataFrame>("input");
  cv::Mat image = md_frame->GetOriginalImage();
  if (image.channels() == 3) {
    cv::cvtColor(image, gray_image_, cv::COLOR_BGR2GRAY);
  } else {
    gray_image_ = image;
  }

  std::vector<Rect> tracked_bboxes;
  if (md_frame->GetBitset().test(MetadataFrame::Bit_bboxes)) {
    std::vector<Rect> bboxes = md_frame->GetBboxes();
    LOG(INFO) << "Got new MetadataFrame, bboxes size is " << bboxes.size()
              << ", current tracker size is " << tracker_list_.size();
    std::vector<Rect> untracked_bboxes = bboxes;
    for (auto it = tracker_list_.begin(); it != tracker_list_.end(); ) {
      (*it)->Track(gray_image_);
      struck::IntRect r((*it)->GetBB());
      cv::Rect rt(r.XMin(), r.YMin(), r.Width(), r.Height());
      float best_percent = 0.f;
      for (auto u_it = untracked_bboxes.begin(); u_it != untracked_bboxes.end(); ++u_it) {
        cv::Rect ru(u_it->px, u_it->py, u_it->width, u_it->height);
        cv::Rect intersects = rt&ru;
        float percent = (float)intersects.area() / (float)ru.area();
        if (percent >= 0.7) {
          untracked_bboxes.erase(u_it);
          best_percent = percent;
          break;
        }
      }
      if (best_percent >= 0.7) {
        tracked_bboxes.push_back(Rect(r.XMin(), r.YMin(), r.Width(), r.Height()));
        it++;
      } else {
        LOG(INFO) << "Remove tracker, best_percent is " << best_percent;
        tracker_list_.erase(it++);
      }
    }

    for (const auto& m: untracked_bboxes) {
      LOG(INFO) << "Create new tracker";
      struck::FloatRect initBB = struck::IntRect(m.px, m.py, m.width, m.height);
      std::shared_ptr<struck::Tracker> new_tracker(new struck::Tracker(conf_));
      new_tracker->Initialise(gray_image_, initBB);
      CHECK(new_tracker->IsInitialised());
      //printf("%s, %d\n", __FUNCTION__, __LINE__);
      new_tracker->Track(gray_image_);
      //printf("%s, %d\n", __FUNCTION__, __LINE__);
      struck::IntRect r(new_tracker->GetBB());
      tracked_bboxes.push_back(Rect(r.XMin(), r.YMin(), r.Width(), r.Height()));
      tracker_list_.push_back(new_tracker);
    }
  } else {
    for (auto it = tracker_list_.begin(); it != tracker_list_.end(); ++it) {
      (*it)->Track(gray_image_);
      struck::IntRect r((*it)->GetBB());
      tracked_bboxes.push_back(Rect(r.XMin(), r.YMin(), r.Width(), r.Height()));
    }
  }

  if (md_frame->GetBitset().test(MetadataFrame::Bit_face_landmarks))
    PushFrame("output", new MetadataFrame(tracked_bboxes, md_frame->GetFaceLandmarks(),
        md_frame->GetOriginalImage()));
  else
    PushFrame("output", new MetadataFrame(tracked_bboxes, md_frame->GetOriginalImage()));

  //LOG(INFO) << "StruckTracker took " << timer.ElapsedMSec() << " ms";
}

ProcessorType StruckTracker::GetType() {
  return PROCESSOR_TYPE_STRUCK_TRACKER;
}
