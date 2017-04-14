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
  std::vector<FaceInfo> faceInfo = md_frame->GetFaceInfo();
  for(int i = 0;i<faceInfo.size();i++){
    float x = faceInfo[i].bbox.x1;
    float y = faceInfo[i].bbox.y1;
    float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 +1;
    float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 +1;
    cv::rectangle(image,cv::Rect(y,x,w,h),cv::Scalar(255,0,0),5);
  }
  for(int i=0;i<faceInfo.size();i++){
    FacePts facePts = faceInfo[i].facePts;
    for(int j=0;j<5;j++)
      cv::circle(image,cv::Point(facePts.y[j],facePts.x[j]),1,cv::Scalar(255,255,0),5);
  }

  std::vector<PointFeature> point_features;
  std::vector<std::vector<float>> face_features = md_frame->GetFaceFeatures();
  CHECK(faceInfo.size() == face_features.size());
  for(int i = 0;i<faceInfo.size();i++){
    cv::Point point((faceInfo[i].bbox.y1 + faceInfo[i].bbox.y2) / 2,
                    (faceInfo[i].bbox.x1 + faceInfo[i].bbox.x2) / 2);
    point_features.push_back(PointFeature(point, face_features[i]));
  }

  if (first_frame_) {
    rem_list_.push_back(point_features);
    first_frame_ = false;
  } else {
    if (rem_list_.size() >= rem_size_) {
      rem_list_.pop_front();
    }
    rem_list_.push_back(point_features);

    auto prev_it=rem_list_.begin();
    for (auto it=rem_list_.begin(); it != rem_list_.end(); ++it) {
      if (it != rem_list_.begin()) {
        const auto& prev_point_features = *prev_it;
        for (const auto& m: *it) {
          auto prev = FindPreviousNearest(m, prev_point_features, 20.0);
          if (prev)
            cv::line(image, prev->point, m.point, cv::Scalar(255,0,0), 5);
        }
      }
      prev_it = it;
    }
  }
  
  PushFrame("output", new ImageFrame(image, md_frame->GetOriginalImage()));
}

ProcessorType ObjectTracker::GetType() {
  return PROCESSOR_TYPE_OBJECT_TRACKER;
}

boost::optional<PointFeature> ObjectTracker::FindPreviousNearest(const PointFeature& point_feature,
                                                                 std::vector<PointFeature> point_features,
                                                                 float threshold)
{
  boost::optional<PointFeature> result;
  float distance = std::numeric_limits<float>::max();
  //printf("=====================FindPreviousNearest====================\n");
  for (const auto& m: point_features) {
    float d = GetDistance(point_feature.face_feature, m.face_feature);
    //printf("%f ", d);
    if ((d < distance) && (d < threshold)) {
      distance = d;
      result = m;
    }
  }
  //printf("\n");
  return result;
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
