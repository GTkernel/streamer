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

#include "frame_receiver.h"

#include <boost/archive/binary_iarchive.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>

constexpr auto SINK = "output";

FrameReceiver::FrameReceiver(const std::string listen_url)
    : Processor(PROCESSOR_TYPE_FRAME_RECEIVER, {}, {SINK}),
      listen_url_(listen_url) {
    deserialize_latency_ms_sum_ = 0;
    received_frame_byte_ = 0;
}

double FrameReceiver::GetMsgByte(){ return received_frame_byte_; }
double FrameReceiver::GetTotalDeserialLatencyMs(){ return deserialize_latency_ms_sum_; }

void FrameReceiver::RunServer(const std::string listen_url) {
  grpc::ServerBuilder builder;

  // Increase maximum message size to 10 MiB
  builder.SetMaxMessageSize(200 * 1024 * 1024);

  // TODO:  Use secure credentials (e.g., SslCredentials)
  builder.AddListeningPort(listen_url, grpc::InsecureServerCredentials());
  builder.RegisterService(this);

  server_ = builder.BuildAndStart();
  LOG(INFO) << "gRPC server started at " << listen_url;
  server_->Wait();
}

StreamPtr FrameReceiver::GetSink() { return Processor::GetSink(SINK); }

grpc::Status FrameReceiver::SendFrame(grpc::ServerContext*,
                                      const SingleFrame* frame_message,
                                      google::protobuf::Empty*) {
  boost::posix_time::ptime start_deserial_time_ = boost::posix_time::microsec_clock::local_time();

  if(received_frame_byte_ == 0){ received_frame_byte_ = frame_message->ByteSize(); }

  std::stringstream frame_string;
  frame_string << frame_message->frame();

  std::unique_ptr<Frame> frame;

  // If Boost's serialization fails, it throws an exception.  If we don't
  // catch the exception, gRPC triggers a core dump with a "double free or
  // corruption" error. See https://github.com/grpc/grpc/issues/3071
  try {
    boost::archive::binary_iarchive ar(frame_string);
    ar >> frame;
  } catch (const boost::archive::archive_exception& e) {
    std::ostringstream error_message;
    error_message << "Boost serialization error: " << e.what();
    LOG(INFO) << error_message.str();
    return grpc::Status(grpc::StatusCode::ABORTED, error_message.str());
  }

  PushFrame(SINK, std::move(frame));

  deserialize_latency_ms_sum_ += (double)(boost::posix_time::microsec_clock::local_time() - start_deserial_time_).total_microseconds();

  return grpc::Status::OK;
}

std::shared_ptr<FrameReceiver> FrameReceiver::Create(
    const FactoryParamsType& params) {
  return std::make_shared<FrameReceiver>(params.at("listen_url"));
}

bool FrameReceiver::Init() {
  std::thread server_thread([this]() { this->RunServer(this->listen_url_); });
  server_thread.detach();
  return true;
}

bool FrameReceiver::OnStop() {
  server_->Shutdown();
  return true;
}

void FrameReceiver::Process() {
  // Frame is already put in the sink of this processor, no need for
  // process to do anything.
}
