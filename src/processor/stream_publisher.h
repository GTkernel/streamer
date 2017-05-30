//
// Created by Ran Xian (xranthoar@gmail.com) on 11/9/16.
//

#ifndef STREAMER_STREAMPUBLISHER_H
#define STREAMER_STREAMPUBLISHER_H

#include "common/common.h"
#include "processor.h"

#include <cppzmq/zmq.hpp>

/**
 * @brief A class for publising streamer's stream to network using ZMQ.
 */
class StreamPublisher : public Processor {
 public:
  StreamPublisher(const string &topic_name);
  virtual ProcessorType GetType() const override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string topic_name_;

  zmq::socket_t &zmq_publisher_;
};

#endif  // STREAMER_STREAMPUBLISHER_H
