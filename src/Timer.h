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
   * \brief Stop the timer.
   */
  void Stop() {
    stop_time_ = std::chrono::system_clock::now();
  }

  double ElaspedMsec() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_ - start_time_).count();
  }

  double ElaspedSec() {
    return std::chrono::duration_cast<std::chrono::seconds>(stop_time_ - start_time_).count();
  }

  double ElapsedMicroSec() {
    return std::chrono::duration_cast<std::chrono::microseconds>(stop_time_ - start_time_).count();
  }

private:
  TimerTimePoint start_time_;
  TimerTimePoint stop_time_;
};

#endif //TX1DNN_TIMER_H
