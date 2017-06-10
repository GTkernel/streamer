//
// Created by Ran Xian (xranthoar@gmail.com) on 11/9/16.
//

#ifndef STREAMER_PROCESSOR_STREAM_PUBLISHER_H_
#define STREAMER_PROCESSOR_STREAM_PUBLISHER_H_

#include "common/common.h"
#include "processor.h"

#include <cppzmq/zmq.hpp>

static const std::string DEFAULT_ZMQ_LISTEN_URL = "127.0.0.1:5536";

/**
 * @brief A class for publising streamer's stream to network using ZMQ.
 */
class StreamPublisher : public Processor {
 public:
  StreamPublisher(const std::string topic_name,
                  const std::string listen_url = DEFAULT_ZMQ_LISTEN_URL);
  ~StreamPublisher();

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string topic_name_;
  zmq::context_t zmq_context_;
  zmq::socket_t zmq_publisher_;
  string zmq_publisher_addr_;
  std::string zmq_listen_url_;
};

#endif  // STREAMER_PROCESSOR_STREAM_PUBLISHER_H_
