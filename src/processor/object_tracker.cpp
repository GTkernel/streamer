/**
* Multi-face tracking using face feature
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#include "common/context.h"
#include "object_tracker.h"

ObjectTracker::ObjectTracker(size_t rem_size)
    : Processor({"input"}, {"output"}),
      rem_size_(rem_size) ,
      first_frame_(true) {}

bool ObjectTracker::Init() {
  LOG(INFO) << "ObjectTracker initialized";
  return true;
}

bool ObjectTracker::OnStop() {
  return true;
}

void ObjectTracker::Process() {
  auto md_frame = GetFrame<MetadataFrame>("input");
  cv::Mat image = md_frame->GetOriginalImage();
  std::vector<Rect> bboxes = md_frame->GetBboxes();
  std::vector<std::vector<float>> face_features = md_frame->GetFaceFeatures();
  CHECK(bboxes.size() == face_features.size());
  
  std::vector<PointFeature> point_features;
  for(int i = 0;i<bboxes.size();i++){
    cv::Point point(bboxes[i].px + bboxes[i].width/2,
                    bboxes[i].py + bboxes[i].height/2);
    point_features.push_back(PointFeature(point, face_features[i]));
  }

  if (first_frame_) {
    first_frame_ = false;
  } else {
    AttachNearest(point_features, 20.0);
  }
  for (const auto& m: point_features) {
    std::list<boost::optional<PointFeature>> l;
    l.push_back(m);
    path_list_.push_back(l);
  }

  for (auto it = path_list_.begin(); it != path_list_.end(); ) {
    if (it->size() > rem_size_)
      it->pop_front();

    bool list_all_empty_point = true;
    for (const auto& m: *it) {
      if (m) list_all_empty_point = false;
    }

    if (list_all_empty_point)
      path_list_.erase(it++);
    else
      it++;
  }

  md_frame->SetPaths(path_list_);
  PushFrame("output", md_frame);
}

ProcessorType ObjectTracker::GetType() {
  return PROCESSOR_TYPE_OBJECT_TRACKER;
}

void ObjectTracker::AttachNearest(std::vector<PointFeature>& point_features,
                                  float threshold)
{
  for (auto& m: path_list_) {
    boost::optional<PointFeature> lp = m.back();
    if (!lp) {
      m.push_back(boost::optional<PointFeature>());
      continue;
    }
    
    auto it_result = point_features.end();
    float distance = std::numeric_limits<float>::max();
    //printf("=====================AttachNearest====================\n");
    for (auto it = point_features.begin(); it != point_features.end(); it++) {
      float d = GetDistance(lp->face_feature, it->face_feature);
      //printf("%f ", d);
      if ((d < distance) && (d < threshold)) {
        distance = d;
        it_result = it;
      }
    }
    //printf("\n");

    if (it_result != point_features.end()) {
      m.push_back(*it_result);
      point_features.erase(it_result);
    } else {
      m.push_back(boost::optional<PointFeature>());
    }
  }
}

float ObjectTracker::GetDistance(const std::vector<float>& a, const std::vector<float>& b)
{
  float distance = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    distance += pow((a[i] - b[i]),2);
  }
  distance = sqrt(distance);

  return distance;
}
