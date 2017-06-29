//
// Created by Ran Xian (xranthoar@gmail.com) on 11/10/16.
//

#include <gtest/gtest.h>
#include <zguide/examples/C++/zhelpers.hpp>

#include "processor/stream_publisher.h"
#include "utils/utils.h"

TEST(STREAM_PUBLISHER_TEST, TEST_PUBLISH_MDSTREAM) {
  StreamPtr stream(new Stream);

  // Publisher
  string publisher_name = "test_publisher";
  StreamPublisher publisher(publisher_name);
  publisher.SetSource("input", stream);

  // Subscriber
  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  subscriber.connect("tcp://localhost:5536");
  if (subscriber.connected()) {
    LOG(INFO) << "Connected";
  } else {
    LOG(INFO) << "Can't connect";
  }
  subscriber.setsockopt(ZMQ_SUBSCRIBE, publisher_name.data(),
                        publisher_name.length());
  STREAMER_SLEEP(100);

  publisher.Start();

  // Push MD frame with tags
  std::vector<string> tags = {"cow", "car"};
  auto frame = std::make_unique<Frame>();
  frame->SetValue("tags", tags);
  stream->PushFrame(std::move(frame));

  LOG(INFO) << "Try to receive";
  string topic_name = s_recv(subscriber);
  EXPECT_EQ("test_publisher", topic_name);
  string content = s_recv(subscriber);
  EXPECT_TRUE(content.size() != 0);

  publisher.Stop();
}
