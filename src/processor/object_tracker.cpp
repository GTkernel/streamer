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
  auto tags = md_frame->GetTags();
  auto boxes = md_frame->GetBboxes();
  CHECK(tags.size() == boxes.size());
  cv::Scalar box_color(255, 0, 0);
  for (size_t i = 0; i < boxes.size(); ++i) {
    cv::Rect rect(boxes[i].px, boxes[i].py, boxes[i].width, boxes[i].height);
    cv::rectangle(image, rect, box_color, 5);
    cv::putText(image, tags[i] , cv::Point(boxes[i].px,boxes[i].py+30) , 0 , 1.0 , cv::Scalar(0,255,0), 3 );
  }

  std::map<std::string, std::vector<cv::Point>> tags_name_points;
  for (size_t i = 0; i < boxes.size(); ++i) {
    cv::Point point(boxes[i].px+(boxes[i].width/2), boxes[i].py+(boxes[i].height/2));
    std::string needle("  :  ");
    std::string::size_type found = tags[i].find(needle);
    CHECK(found!=std::string::npos) << "Can not find " << needle << " in " << tags[i];
    std::string tags_name = tags[i].substr(0, found-0);
    tags_name_points[tags_name].push_back(point);
  }

  if (first_frame_) {
    rem_list_.push_back(tags_name_points);
    first_frame_ = false;
  } else {
    if (rem_list_.size() >= rem_size_) {
      rem_list_.pop_front();
    }
    rem_list_.push_back(tags_name_points);

    auto prev_it=rem_list_.begin();
    for (auto it=rem_list_.begin(); it != rem_list_.end(); ++it) {
      if (it != rem_list_.begin()) {
        for (const auto& m_pair: *it) {
          for (const auto& m_point: m_pair.second) {
            auto prev_map = *prev_it;
            if (prev_map.find(m_pair.first) != prev_map.end()) {
              cv::Point prev_point = FindPreviousNearest(m_point, prev_map[m_pair.first]);
              cv::line(image, prev_point, m_point, cv::Scalar(0,255,0), 3);
            }
          }
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

cv::Point ObjectTracker::FindPreviousNearest(const cv::Point& point, const std::vector<cv::Point>& points)
{
  CHECK(points.size() > 0);
  cv::Point result;
  int distance = std::numeric_limits<int>::max();
  for (const auto& m: points) {
    int d = GetDistance(point, m);
    if (d < distance) {
      distance = d;
      result = m;
    }
  }
  return result;
}

int ObjectTracker::GetDistance(const cv::Point& a, const cv::Point& b)
{
  int distance;
  distance = pow((a.x - b.x),2) + pow((a.y - b.y),2);
  distance = sqrt(distance);

  return distance;
}
