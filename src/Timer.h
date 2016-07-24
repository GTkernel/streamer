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
  /**
   * \brief Get current system time point.
   * @return
   */
  static inline TimerTimePoint GetCurrentTime() {
    std::chrono::system_clock::now();
    return std::chrono::system_clock::now();
  }
  /**
   * \brief Get difference between two time points in mircoseconds.
   * @param end_time
   * @param start_time
   * @return
   */
  static inline double GetTimeDiffMicroSeconds(
    const TimerTimePoint &begin_time,
    const TimerTimePoint &end_time) {

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time);
    return elapsed.count();
  }
};

#endif //TX1DNN_TIMER_H
