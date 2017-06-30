#include "frame_publisher.h"

#include <boost/archive/binary_oarchive.hpp>
#include <zguide/examples/C++/zhelpers.hpp>

constexpr auto SOURCE = "input";

FramePublisher::FramePublisher(const std::string url)
    : Processor(PROCESSOR_TYPE_FRAME_PUBLISHER, {SOURCE}, {}),
      zmq_context_{1},
      zmq_publisher_{zmq_context_, ZMQ_PUB},
      zmq_publisher_addr_("tcp://" + url) {
  // Bind the publisher socket
  LOG(INFO) << "Publishing frames on " << zmq_publisher_addr_;
  try {
    zmq_publisher_.bind(zmq_publisher_addr_);
  } catch (const zmq::error_t& e) {
    LOG(FATAL) << "ZMQ bind error: " << e.what();
  }
}

FramePublisher::~FramePublisher() {
  // Tear down the publisher socket
  zmq_publisher_.unbind(zmq_publisher_addr_);
}

void FramePublisher::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE, stream);
}

std::shared_ptr<FramePublisher> FramePublisher::Create(
    const FactoryParamsType& params) {
  if (params.count("url") != 0) {
    return std::make_shared<FramePublisher>(params.at("url"));
  }
  return std::make_shared<FramePublisher>();
}

bool FramePublisher::Init() { return true; }

bool FramePublisher::OnStop() { return true; }

void FramePublisher::Process() {
  auto frame = this->GetFrame(SOURCE);

  // serialize
  std::stringstream frame_string;
  try {
    boost::archive::binary_oarchive ar(frame_string);
    ar << frame;
  } catch (const boost::archive::archive_exception& e) {
    LOG(INFO) << "Boost serialization error: " << e.what();
  }

  s_send(zmq_publisher_, frame_string.str());
}
