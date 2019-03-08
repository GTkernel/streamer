// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frame_sender.h"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

constexpr auto SOURCE = "input";
constexpr auto SINK_NAME = "output";

FrameSender::FrameSender(const std::string server_url,
                         std::unordered_set<std::string> fields_to_send)
    : Processor(PROCESSOR_TYPE_FRAME_SENDER, {SOURCE}, {SINK_NAME}),
      server_url_(server_url),
      fields_to_send_(fields_to_send) {
  auto channel =
      grpc::CreateChannel(server_url_, grpc::InsecureChannelCredentials());
  stub_ = Messenger::NewStub(channel);
  serialize_latency_ms_sum = 0;
  send_latency_ms_sum = 0;
}

void FrameSender::SetSource(StreamPtr stream) {
  Processor::SetSource(SOURCE, stream);
}

StreamPtr FrameSender::GetSink() {
  return Processor::GetSink(SINK_NAME);
}

std::shared_ptr<FrameSender> FrameSender::Create(
    const FactoryParamsType& params) {
  return std::make_shared<FrameSender>(params.at("server_url"));
}

bool FrameSender::Init() { return true; }

bool FrameSender::OnStop() { return true; }

void FrameSender::Process() {
  auto frame = this->GetFrame(SOURCE);

  // serialize
  std::stringstream frame_string;
  
  boost::posix_time::ptime start_serial_time_ = boost::posix_time::microsec_clock::local_time();
  try {
    boost::archive::binary_oarchive ar(frame_string);
    auto copy_frame = std::make_unique<Frame>(frame, fields_to_send_);
    ar << copy_frame;
  } catch (const boost::archive::archive_exception& e) {
    LOG(INFO) << "Boost serialization error: " << e.what();
  }

  //save the serialize time in local
  serialize_latency_ms_sum +=
        (double)(boost::posix_time::microsec_clock::local_time() - start_serial_time_).total_microseconds();

  SingleFrame frame_message;
  frame_message.set_frame(frame_string.str());

  grpc::ClientContext context;
  google::protobuf::Empty ignored;

  boost::posix_time::ptime start_send_time_ = boost::posix_time::microsec_clock::local_time();

  grpc::Status status = stub_->SendFrame(&context, frame_message, &ignored);
  send_latency_ms_sum +=
        (double)(boost::posix_time::microsec_clock::local_time() - start_send_time_).total_microseconds();

  if (!status.ok()) {
    LOG(INFO) << "gRPC error(SendFrame): " << status.error_message();
  }
}
