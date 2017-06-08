//
// Created by Ran Xian (xranthoar@gmail.com) on 11/9/16.
//

#ifndef STREAMER_STREAMPUBLISHER_H
#define STREAMER_STREAMPUBLISHER_H

#include "common/common.h"
#include "processor.h"

#include <cppzmq/zmq.hpp>

static const std::string DEFAULT_PUBLISHER_ADDRESS = "127.0.0.1";
static const unsigned int DEFAULT_PUBLISHER_PORT = 5536;

/**
 * @brief A class for publising streamer's stream to network using ZMQ.
 */
class StreamPublisher : public Processor {
 public:
  StreamPublisher(const string &topic_name,
                  const std::string address = DEFAULT_PUBLISHER_ADDRESS,
                  const unsigned int port = DEFAULT_PUBLISHER_PORT);
  ~StreamPublisher();
  virtual ProcessorType GetType() const override;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string topic_name_;
  zmq::context_t zmq_context_;
  zmq::socket_t zmq_publisher_;
  string zmq_publisher_addr_;
  std::string zmq_address_;
  unsigned int zmq_port_;
};

#endif  // STREAMER_STREAMPUBLISHER_H
