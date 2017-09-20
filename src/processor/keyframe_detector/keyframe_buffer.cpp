
#include "processor/keyframe_detector/keyframe_buffer.h"

#include <cmath>
#include <sstream>

#include <opencv2/opencv.hpp>

KeyframeBuffer::KeyframeBuffer(float sel, size_t buf_len)
    : target_buf_len_(buf_len), on_first_buf_(true) {
  SetSelectivity(sel);
  // Allocate extra space for the last keyframe from the previous buffer.
  buf_.reserve(buf_len + 1);
}

void KeyframeBuffer::Stop() {
  if (log_.is_open()) {
    log_.close();
  }
}

void KeyframeBuffer::SetSelectivity(float new_sel) {
  CHECK(new_sel > 0 && new_sel <= 1)
      << "Selectivity must be in the range (0, 1], but it is: " << new_sel;
  sel_ = new_sel;
}

void KeyframeBuffer::EnableLog(std::string output_dir,
                               std::string output_prefix) {
  std::ostringstream filepath;
  filepath << output_dir << "/" << output_prefix << "_" << sel_ << "_"
           << target_buf_len_ << ".log";
  log_.open(filepath.str());
}

std::vector<std::unique_ptr<Frame>> KeyframeBuffer::Push(
    std::unique_ptr<Frame> frame) {
  buf_.push_back(std::move(frame));

  idx_t target_buf_len_actual = target_buf_len_;
  // If we're on the first buffer, then we're targeting a buffer length of
  // one less than the total size of the buffer.
  if (!on_first_buf_) {
    ++target_buf_len_actual;
  }

  std::vector<std::unique_ptr<Frame>> keyframes;
  if (buf_.size() == target_buf_len_actual) {
    // The frame buffer is full, meaning that it's time to find the keyframes.
    unsigned long start_frame_id;
    if (on_first_buf_) {
      start_frame_id = buf_.front()->GetValue<unsigned long>("frame_id");
    } else {
      start_frame_id = buf_[1]->GetValue<unsigned long>("frame_id");
    }
    unsigned long end_frame_id =
        buf_.back()->GetValue<unsigned long>("frame_id");
    LOG(INFO) << "Keyframe detection running over frame range: {"
              << start_frame_id << ", " << end_frame_id << "}";

    std::vector<idx_t> keyframe_idxs = GetKeyframeIdxs();
    auto idxs_it = keyframe_idxs.begin();

    if (on_first_buf_) {
      on_first_buf_ = false;
    } else {
      // If this is not the first buffer, then the first keyframe is actually
      // the last keyframe selected from the previous buffer, so we need to
      // discard it.
      ++idxs_it;
    }

    for (; idxs_it != keyframe_idxs.end(); ++idxs_it) {
      std::unique_ptr<Frame> keyframe = std::move(buf_.at(*idxs_it));
      if (log_.is_open()) {
        log_ << keyframe->GetValue<unsigned long>("frame_id") << "\n";
      }
      keyframes.push_back(std::move(keyframe));
    }

    // Reset the frame buffer. Push the last keyframe onto the frame buffer. It
    // will be the first frame in the next buffer's run of the algorithm.
    buf_.clear();
    buf_.push_back(std::make_unique<Frame>(keyframes.back()));
  }
  return keyframes;
}

std::vector<KeyframeBuffer::idx_t> KeyframeBuffer::GetKeyframeIdxs() const {
  // Boost L2 norm:
  // http://www.boost.org/doc/libs/1_51_0/libs/numeric/ublas/doc/operations_overview.htm
  // Eigen L2 norm:
  // https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html

  if (sel_ > 1) {
    // We are trying to select more keyframes than there are frames in our
    // frame buffer. This should never happen, because "sel_" is in the range
    // (0, 1].
    LOG(FATAL) << "Selectivity must be in the range (0, 1], but it is: "
               << sel_;
  } else if (sel_ == 1) {
    // We are trying to find as many keyframes as there are frames in our
    // frame buffer, so we'll return all of the frames' indices.
    std::vector<idx_t> keyframe_idxs;
    for (idx_t i = 0; i < buf_.size(); ++i) {
      keyframe_idxs.push_back(i);
    }
    return keyframe_idxs;
  }

  // Run the longest paths algorithm. We are interested in the longest *k-step*
  // path in a DAG (that contains more than (k + 1) frames), so we cannot use a
  // traditional longest (or shorted) paths algorithm.

  idx_t num_frames = buf_.size();
  idx_t num_frames_in_path = (idx_t)ceil(sel_ * num_frames);
  if (!on_first_buf_) {
    // Add one to account for the fact that the first frame in the buffer is the
    // last keyframe from the previous buffer.
    ++num_frames_in_path;
  }
  // The number of graph edges that must be traversed by our path is one less
  // than the number of nodes in the path.
  idx_t num_steps = num_frames_in_path - 1;

  // Matrix where entry (i, j) is the Euclidean distance between frames i and j.
  // Only entries where j > i are populated (i.e. only the distances from each
  // frame to every future frame).
  std::vector<std::vector<double>> direct_lens(num_frames,
                                               std::vector<double>(num_frames));

  auto start_id = buf_.front()->GetValue<unsigned long>("frame_id");
  auto end_id = buf_.back()->GetValue<unsigned long>("frame_id");
  auto span = (end_id - start_id) / 4;

  // TODO: This should be optimized.
  for (idx_t i = 0; i < num_frames; ++i) {
    const cv::Mat& src_f = buf_.at(i)->GetValue<cv::Mat>("activations");
    auto i_id = buf_.at(i)->GetValue<unsigned long>("frame_id");

    for (idx_t j = i + 1; j < num_frames; ++j) {
      const cv::Mat& dst_f = buf_.at(j)->GetValue<cv::Mat>("activations");
      auto j_id = buf_.at(j)->GetValue<unsigned long>("frame_id");

      direct_lens[i][j] = cv::norm(dst_f - src_f);  // / (span + j_id - i_id);
    }
  }

  // Matrix where entry (k, i) is a pair where the first entry is the previous
  // frame in the longest k-step path from frame 0 to frame i and the second
  // entry is the length of the longest k-step path from frame 0 to frame i.
  std::vector<std::vector<std::pair<idx_t, double>>> path_info(
      num_steps + 1, std::vector<std::pair<idx_t, double>>(num_frames));

  // The lens of 1-step paths are just the direct lengths from the first frame
  // to each other frame.
  for (idx_t i = 0; i < num_frames; ++i) {
    path_info[1][i] = {0, direct_lens.at(0).at(i)};
  }

  // Loop over remaining path lengths.
  for (idx_t k = 2; k < num_steps + 1; ++k) {
    // The longest (k - 1)-step path to each frame.
    std::vector<std::pair<idx_t, double>> k_minus_1_paths = path_info.at(k - 1);
    // The longest k-step path to each frame.
    std::vector<std::pair<idx_t, double>> k_paths = path_info.at(k);

    // For each frame (i.e. possible path end frame), find the longest k-step
    // path from the first frame to that frame. "end_f" is the index of a
    // frame at which the path ends.
    for (idx_t end_f = 0; end_f < num_frames; ++end_f) {
      if (end_f < k) {
        // No path exists. Paths of length k require at least (k + 1) nodes.
        k_paths[end_f].second = 0;
      } else {
        double max_len = 0;
        // Find the longest path to the end frame. Valid intermediate frames are
        // the frames from (k - 1) (because k frames are required to make a
        // (k - 1)-step path) until (end_f - 1). "int_f" is the index of an
        // intermediate frame.
        for (idx_t int_f = k - 1; int_f < end_f; ++int_f) {
          // Find the intermediate frame that produces the longest k-step path
          // to the the end frame. We're finding the k-step path because we're
          // starting with the longest (k - 1)-step path to each intermediate
          // frame, and then there is a 1-step path from the intermediate frame
          // to the end frame.
          double len = k_minus_1_paths.at(int_f).second +
                       direct_lens.at(int_f).at(end_f);
          if (len > max_len) {
            max_len = len;
            // Record that this intermediate node gives the longest k-step path
            // to the end frame.
            k_paths[end_f].first = int_f;
          }
        }

        // Record that the longest k-step path to the end frame has length
        // max_len.
        k_paths[end_f].second = max_len;
      }
    }
    path_info[k] = k_paths;
  }

  // Determine which frame is the destination frame (i.e. the frame with the
  // longest k-step path from the first frame).
  double max_len = 0;
  idx_t end_f = 0;
  std::vector<std::pair<idx_t, double>> final_path = path_info[num_steps];
  for (idx_t i = 0; i < num_frames; ++i) {
    double len = final_path.at(i).second;
    if (len > max_len) {
      max_len = len;
      end_f = i;
    }
  }

  // Extract from the previous_frames matrix the indices of the frames that make
  // up the full path. We work backwards from "end_f", the last frame in the
  // path.
  std::vector<idx_t> keyframe_idxs(num_frames_in_path);
  keyframe_idxs[num_frames_in_path - 1] = end_f;
  idx_t last_f = end_f;
  for (int k = num_frames_in_path - 1; k > 0; --k) {
    idx_t prev_f = path_info.at(k).at(last_f).first;
    keyframe_idxs[k - 1] = prev_f;
    last_f = prev_f;
  }

  return keyframe_idxs;
}
