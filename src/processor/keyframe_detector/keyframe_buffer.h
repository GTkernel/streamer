
#ifndef STREAMER_PROCESSOR_KEYFRAME_DETECTOR_KEYFRAME_BUFFER_H_
#define STREAMER_PROCESSOR_KEYFRAME_DETECTOR_KEYFRAME_BUFFER_H_

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "stream/frame.h"

// This class encapsulates the logic for selecting keyframes from incoming
// frames. New frames are registered using the "Push()" function. When the
// the internal buffer has filled to its user-defined threshold, "Push()" will
// return the keyframes selected from the buffer.
//
// A KeyframeBuffer is parameterized by two values: its selectivity, which
// denotes the target decimation ratio, and its buffer's length, which
// determines when keyframe detection is run. The selectivity can be changed at
// any time using "SetSelectivity()", but the buffer length is fixed.
class KeyframeBuffer {
 public:
  // "sel" is a selectivity in the range (0, 1] and "buf_len" is the buffer
  // length at which keyframe detection will be triggered.
  KeyframeBuffer(float sel, size_t buf_len);
  // A KeyframeBuffer object contains unique pointers to Frames, so it cannot be
  // copied.
  KeyframeBuffer(const KeyframeBuffer&) = delete;
  KeyframeBuffer& operator=(const KeyframeBuffer&) = delete;

  // Closes the log file.
  void Stop();
  // "new_sel" must be in the range (0, 1].
  void SetSelectivity(float new_sel);
  // Signals the KeyframeBuffer to begin logging the "frame_id" field of each
  // keyframe to a file. The file will be written to the directory "output_dir"
  // and will be named "<output_prefix>_<selectivity>_<buffer length>.log".
  void EnableLog(std::string output_dir, std::string output_prefix);
  // Adds the provided frame to the interval frame buffer. Normally, the return
  // value is an empty vector. If adding the provided frame makes the internal
  // frame buffer exceed its threshold, then keyframe detection is triggered
  // and the selected keyframes are returned.
  std::vector<std::unique_ptr<Frame>> Push(std::unique_ptr<Frame> frame);

 private:
  std::vector<std::unique_ptr<Frame>> buf_;
  // This type is used to index into "buf_" and represent frames while running
  // the keyframe detection algorithm.
  typedef decltype(buf_.size()) idx_t;

  // Returns the indices of the keyframes in the current frame buffer. Does not
  // modify the frame buffer.
  std::vector<idx_t> GetKeyframeIdxs() const;

  float sel_;
  // The index of the next empty slot in the frame buffer.
  idx_t buf_idx_;
  // True if the current frame buffer is the first frame buffer (i.e. keyframe
  // detection has never been run). This is a special case, since the target
  // buffer length will be one less than normal. This is because in steady-state
  // operation, we preload the buffer with the last keyframe from the last run,
  // so we need the buffer to have one extra slot.
  bool on_first_buf_;
  // If this stream is open, then the keyframe selection results will be written
  // to it.
  std::ofstream log_;
};

#endif  // STREAMER_PROCESSOR_KEYFRAME_DETECTOR_KEYFRAME_BUFFER_H_
