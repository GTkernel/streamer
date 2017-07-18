//
// Created by Ran Xian (xranthoar@gmail.com) on 10/9/16.
//

#include "frame.h"

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

Frame::Frame(double start_time) { frame_data_["start_time_ms"] = start_time; }

Frame::Frame(const std::unique_ptr<Frame>& frame) : Frame(*frame.get()) {}

Frame::Frame(const Frame& frame) {
  frame_data_ = frame.frame_data_;
  // Deep copy the original bytes
  auto it = frame.frame_data_.find("original_bytes");
  if (it != frame.frame_data_.end()) {
    std::vector<char> newbuf(boost::get<std::vector<char>>(it->second));
    frame_data_["original_bytes"] = newbuf;
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

// Types declared in field_types of frame
template void Frame::SetValue(std::string, const double&);
template void Frame::SetValue(std::string, const float&);
template void Frame::SetValue(std::string, const int&);
template void Frame::SetValue(std::string, const unsigned long&);
template void Frame::SetValue(std::string, const std::string&);
template void Frame::SetValue(std::string, const std::vector<std::string>&);
template void Frame::SetValue(std::string, const std::vector<Rect>&);
template void Frame::SetValue(std::string, const std::vector<char>&);
template void Frame::SetValue(std::string, const cv::Mat&);

template double Frame::GetValue(std::string) const;
template float Frame::GetValue(std::string) const;
template int Frame::GetValue(std::string) const;
template unsigned long Frame::GetValue(std::string) const;
template std::string Frame::GetValue(std::string) const;
template std::vector<std::string> Frame::GetValue(std::string) const;
template std::vector<Rect> Frame::GetValue(std::string) const;
template cv::Mat Frame::GetValue(std::string) const;
template std::vector<char> Frame::GetValue(std::string) const;
