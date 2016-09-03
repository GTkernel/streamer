//
// Created by Ran Xian on 7/23/16.
//

#ifndef TX1DNN_TIMER_H
#define TX1DNN_TIMER_H

#include<chrono>

/**
 * \brief Class used to measure wall clock time.
 */
class Timer {
public:
  typedef std::chrono::time_point<std::chrono::system_clock> TimerTimePoint;

public:
  Timer() {}

  /**
   * \brief Start the timer.
   */
  void Start() {
    start_time_ = std::chrono::system_clock::now();
  }

  /**
   * \brief Get elapsed time in milliseconds.
   */
  double ElapsedMSec() {
    return ElapsedMicroSec() / 1000.0;
  }

  /**
   * \brief Get elapsed time in seconds.
   */
  double ElapsedSec() {
    return ElapsedMicroSec() / 1000000.0;
  }

  /**
   * \brief Get elapsed time in micro seconds.
   */
  double ElapsedMicroSec() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time_).count();
  }
private:
  TimerTimePoint start_time_;
};

#endif //TX1DNN_TIMER_H
