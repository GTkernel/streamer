//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#include "frame.h"

#include <algorithm>
#include <experimental/unordered_map>

#include <opencv2/core/core.hpp>

#include "common/types.h"

class FramePrinter : public boost::static_visitor<std::string> {
 public:
  std::string operator()(const double& v) const {
    std::ostringstream output;
    output << v;
    return output.str();
  }

  std::string operator()(const float& v) const {
    std::ostringstream output;
    output << v;
    return output.str();
  }

  std::string operator()(const int& v) const {
    std::ostringstream output;
    output << v;
    return output.str();
  }

  std::string operator()(const unsigned long& v) const {
    std::ostringstream output;
    output << v;
    return output.str();
  }

  std::string operator()(const boost::posix_time::ptime& v) const {
    return boost::posix_time::to_simple_string(v);
  }

  std::string operator()(const std::string& v) const { return v; }

  std::string operator()(const std::vector<std::string>& v) const {
    std::ostringstream output;
    output << "std::vector<std::string> = [" << std::endl;
    for (auto& s : v) {
      output << s << std::endl;
    }
    output << "]";
    return output.str();
  }

  std::string operator()(const std::vector<double>& v) const {
    std::ostringstream output;
    output << "std::vector<double> = [" << std::endl;
    for (auto& s : v) {
      output << s << std::endl;
    }
    output << "]";
    return output.str();
  }

  std::string operator()(const std::vector<Rect>& v) const {
    std::ostringstream output;
    output << "std::vector<Rect> = [" << std::endl;
    for (auto& r : v) {
      output << "Rect("
             << "px = " << r.px << "py = " << r.py << "width = " << r.width
             << "height = " << r.height << ")" << std::endl;
    }
    output << "]";
    return output.str();
  }

  std::string operator()(const std::vector<char>& v) const {
    std::ostringstream output;
    output << "std::vector<char>(size = " << v.size() << ") = [";
    decltype(v.size()) num_elems = v.size();
    if (num_elems > 3) {
      num_elems = 3;
    }
    for (decltype(num_elems) i = 0; i < num_elems; ++i) {
      output << +v[i] << ", ";
    }
    output << "...]";
    return output.str();
  }

  std::string operator()(const cv::Mat& v) const {
    std::ostringstream output, mout;
    cv::Mat tmp;
    v(cv::Rect(0, 0, 3, 1)).copyTo(tmp);
    mout << tmp;
    output << "cv::Mat(size = " << v.cols << "x" << v.rows
           << ") = " << mout.str().substr(0, 20) << "...]";
    return output.str();
  }
};

class FrameJsonPrinter : public boost::static_visitor<nlohmann::json> {
 public:
  nlohmann::json operator()(const double& v) const { return v; }

  nlohmann::json operator()(const float& v) const { return v; }

  nlohmann::json operator()(const int& v) const { return v; }

  nlohmann::json operator()(const unsigned long& v) const { return v; }

  nlohmann::json operator()(const boost::posix_time::ptime& v) const {
    return boost::posix_time::to_simple_string(v);
  }

  nlohmann::json operator()(const std::string& v) const { return v; }

  nlohmann::json operator()(const std::vector<std::string>& v) const {
    return v;
  }

  nlohmann::json operator()(const std::vector<double>& v) const { return v; }

  nlohmann::json operator()(const std::vector<Rect>& v) const {
    nlohmann::json j;
    for (const auto& r : v) {
      j.push_back(r.ToJson());
    }
    return j;
  }

  nlohmann::json operator()(const std::vector<char>& v) const { return v; }

  nlohmann::json operator()(const cv::Mat& v) const {
    cv::FileStorage fs(".json", cv::FileStorage::WRITE |
                                    cv::FileStorage::MEMORY |
                                    cv::FileStorage::FORMAT_JSON);
    fs << "cvMat" << v;
    return fs.releaseAndGetString();
  }
};

Frame::Frame(double start_time) { frame_data_["start_time_ms"] = start_time; }

Frame::Frame(const std::unique_ptr<Frame>& frame) : Frame(*frame.get()) {}

Frame::Frame(const Frame& frame) : Frame(frame, {}) {}

Frame::Frame(const Frame& frame, std::unordered_set<std::string> fields) {
  frame_data_ = frame.frame_data_;

  bool inherit_all_fields = fields.empty();
  if (!inherit_all_fields) {
    std::experimental::erase_if(frame_data_, [&fields](auto& e) {
      return fields.find(e.first) == fields.end();
    });
  }

  // If either we are inheriting all fields or we are explicitly inheriting
  // "original_bytes", and "original_bytes" is a valid field in "frame", then
  // we need to inherit the "original_bytes" field. Doing so requires a deep
  // copy.
  auto other_it = frame.frame_data_.find("original_bytes");
  auto field_it = std::find(fields.begin(), fields.end(), "original_bytes");
  if ((inherit_all_fields || (field_it != fields.end())) &&
      (other_it != frame.frame_data_.end())) {
    frame_data_["original_bytes"] =
        boost::get<std::vector<char>>(other_it->second);
  }
}

template <typename T>
T Frame::GetValue(std::string key) const {
  auto it = frame_data_.find(key);
  if (it != frame_data_.end()) {
    return boost::get<T>(it->second);
  } else {
    throw std::out_of_range(key);
  }
}

template <typename T>
void Frame::SetValue(std::string key, const T& val) {
  frame_data_[key] = val;
}

std::string Frame::ToString() const {
  FramePrinter visitor;
  std::ostringstream output;
  for (auto iter = frame_data_.begin(); iter != frame_data_.end(); iter++) {
    auto res = boost::apply_visitor(visitor, iter->second);
    output << iter->first << ": " << res << std::endl;
  }
  return output.str();
}

nlohmann::json Frame::ToJson() const {
  FrameJsonPrinter visitor;
  nlohmann::json j;
  for (const auto& e : frame_data_) {
    j[e.first] = boost::apply_visitor(visitor, e.second);
  }
  return j;
}

// Types declared in Field
template void Frame::SetValue(std::string, const double&);
template void Frame::SetValue(std::string, const float&);
template void Frame::SetValue(std::string, const int&);
template void Frame::SetValue(std::string, const unsigned long&);
template void Frame::SetValue(std::string, const boost::posix_time::ptime&);
template void Frame::SetValue(std::string, const std::string&);
template void Frame::SetValue(std::string, const std::vector<std::string>&);
template void Frame::SetValue(std::string, const std::vector<double>&);
template void Frame::SetValue(std::string, const std::vector<Rect>&);
template void Frame::SetValue(std::string, const std::vector<char>&);
template void Frame::SetValue(std::string, const cv::Mat&);

template double Frame::GetValue(std::string) const;
template float Frame::GetValue(std::string) const;
template int Frame::GetValue(std::string) const;
template unsigned long Frame::GetValue(std::string) const;
template boost::posix_time::ptime Frame::GetValue(std::string) const;
template std::string Frame::GetValue(std::string) const;
template std::vector<std::string> Frame::GetValue(std::string) const;
template std::vector<double> Frame::GetValue(std::string) const;
template std::vector<Rect> Frame::GetValue(std::string) const;
template cv::Mat Frame::GetValue(std::string) const;
template std::vector<char> Frame::GetValue(std::string) const;
