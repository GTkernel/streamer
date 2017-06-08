//
// Created by Ran Xian (xranthoar@gmail.com) on 11/9/16.
//

#include "stream_publisher.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cppzmq/zhelper.hpp>

namespace pt = boost::property_tree;

StreamPublisher::StreamPublisher(const string& topic_name,
                                 const std::string address,
                                 const unsigned int port)
    : Processor({"input"}, {}),
      topic_name_(topic_name),
      zmq_context_{1},
      zmq_publisher_{zmq_context_, ZMQ_PUB},
      zmq_address_(address),
      zmq_port_(port) {
  // Bind the publisher socket
  zmq_publisher_addr_ =
      "tcp://" + zmq_address_ + ":" + std::to_string(zmq_port_);
  LOG(INFO) << zmq_publisher_addr_;
  try {
    zmq_publisher_.bind(zmq_publisher_addr_);
  } catch (const zmq::error_t &e) {
    LOG(FATAL) << "ZMQ bind error: " << e.what();
  }
}

StreamPublisher::~StreamPublisher() {
  // Tear down the publisher socket
  zmq_publisher_.unbind(zmq_publisher_addr_);
}

ProcessorType StreamPublisher::GetType() const {
  return PROCESSOR_TYPE_STREAM_PUBLISHER;
}

bool StreamPublisher::Init() { return true; }

bool StreamPublisher::OnStop() { return true; }

void StreamPublisher::Process() {
  auto frame = GetFrame("input");
  switch (frame->GetType()) {
    case FRAME_TYPE_BYTES: {
      STREAMER_NOT_IMPLEMENTED;
      break;
    }
    case FRAME_TYPE_IMAGE: {
      STREAMER_NOT_IMPLEMENTED;
      break;
    }
    case FRAME_TYPE_MD: {
      // TODO: Move the serialization function to the logic of frame class
      s_sendmore(zmq_publisher_, topic_name_);
      auto md_frame = std::dynamic_pointer_cast<MetadataFrame>(frame);

      pt::ptree root;
      pt::ptree tags_node;
      pt::ptree bboxes_node;

      root.put("type", "metadata");

      for (const auto& tag : md_frame->GetTags()) {
        pt::ptree tag_node;
        tag_node.put("", tag);
        tags_node.push_back({"", tag_node});
      }

      for (const auto& bbox : md_frame->GetBboxes()) {
        pt::ptree bbox_node;
        bbox_node.put("x", bbox.px);
        bbox_node.put("y", bbox.py);
        bbox_node.put("width", bbox.width);
        bbox_node.put("height", bbox.height);
        bboxes_node.push_back({"", bbox_node});
      }

      root.add_child("tags", tags_node);
      root.add_child("bboxes", bboxes_node);

      std::ostringstream ss;
      pt::write_json(ss, root);

      s_send(zmq_publisher_, ss.str());

      LOG(INFO) << "Sent";
      break;
    }
    case FRAME_TYPE_INVALID: {
      s_send(zmq_publisher_, topic_name_);
      s_send(zmq_publisher_, "INVALID");
      break;
    }
  }
}
