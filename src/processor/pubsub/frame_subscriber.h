#ifndef STREAMER_PROCESSOR_PUBSUB_FRAME_SUBSCRIBER_H_
#define STREAMER_PROCESSOR_PUBSUB_FRAME_SUBSCRIBER_H_

#include <zmq.hpp>

#include "common/types.h"
#include "processor/processor.h"

static const std::string DEFAULT_ZMQ_SUB_URL = "127.0.0.1:5536";

/**
 * @brief A class for subscribing to a stream using ZMQ.
 */
class FrameSubscriber : public Processor {
 public:
  FrameSubscriber(const std::string url = DEFAULT_ZMQ_SUB_URL);
  ~FrameSubscriber();

  StreamPtr GetSink();
  StreamPtr GetSink(const std::string& name) = delete;

  static std::shared_ptr<FrameSubscriber> Create(
      const FactoryParamsType& params);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  zmq::context_t zmq_context_;
  zmq::socket_t zmq_subscriber_;
};

#endif  // STREAMER_PROCESSOR_PUBSUB_FRAME_SUBSCRIBER_H_
