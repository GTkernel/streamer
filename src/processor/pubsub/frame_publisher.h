#ifndef STREAMER_PROCESSOR_PUBSUB_FRAME_PUBLISHER_H_
#define STREAMER_PROCESSOR_PUBSUB_FRAME_PUBLISHER_H_

#include <zmq.hpp>

#include "common/types.h"
#include "processor/processor.h"

static const std::string DEFAULT_ZMQ_PUB_URL = "127.0.0.1:5536";

/**
 * @brief A class for publishing a stream on the network using ZMQ.
 */
class FramePublisher : public Processor {
 public:
  FramePublisher(const std::string url = DEFAULT_ZMQ_PUB_URL);
  ~FramePublisher();

  void SetSource(StreamPtr stream);
  using Processor::SetSource;

  static std::shared_ptr<FramePublisher> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  zmq::context_t zmq_context_;
  zmq::socket_t zmq_publisher_;
  std::string zmq_publisher_addr_;
};

#endif  // STREAMER_PROCESSOR_PUBSUB_FRAME_PUBLISHER_H_
