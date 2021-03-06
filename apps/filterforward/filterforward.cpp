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

#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <curl/curl.h>
#include <glog/logging.h>
#include <gst/gst.h>
#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "common/context.h"
#include "common/types.h"
#include "model/model_manager.h"
#include "processor/flow_control/flow_control_entrance.h"
#include "processor/flow_control/flow_control_exit.h"
#include "processor/frame_writer.h"
#include "processor/image_transformer.h"
#include "processor/imagematch/imagematch.h"
#include "processor/jpeg_writer.h"
#include "processor/keyframe_detector/keyframe_detector.h"
#include "processor/neural_net_evaluator.h"
#include "processor/processor.h"
#include "processor/pubsub/frame_subscriber.h"
#include "processor/throttler.h"
#include "stream/frame.h"
#include "stream/stream.h"
#include "utils/perf_utils.h"
#include "utils/string_utils.h"

namespace po = boost::program_options;

constexpr auto JPEG_WRITER_FIELD = "original_image";
std::unordered_set<std::string> FRAME_WRITER_FIELDS({"frame_id",
                                                     "capture_time_micros",
                                                     "imagematch.match_prob"});

// Used to signal all threads that the pipeline should stop.
std::atomic<bool> stopped(false);

typedef struct {
  int num;
  std::string layer;
  int xmin;
  int ymin;
  int xmax;
  int ymax;
  bool flat;
  std::string path;
  float threshold;
} query_spec_t;

// Designed to be run in its own thread. Sets "stopped" to true after num_frames
// have been processed or after a stop frame has been detected.
void Stopper(StreamPtr stream, unsigned int num_frames) {
  unsigned int count = 0;
  StreamReader* reader = stream->Subscribe();
  while (!stopped && (num_frames == 0 || ++count < num_frames + 1)) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame != nullptr && frame->IsStopFrame()) {
      break;
    }
  }
  stopped = true;
  reader->UnSubscribe();
}

// Designed to be run in its own thread. Creates a log file containing
// performance metrics for the specified stream.
void Logger(size_t idx, StreamPtr stream, boost::posix_time::ptime log_time,
            std::vector<std::string> fields, const std::string& output_dir,
            bool log_memory, bool display = false) {
  cv::Mat current_image;
  cv::Mat last_match = cv::Mat::zeros(640, 480, CV_32F);
  if (display) {
    cv::namedWindow("current_image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("last_match", cv::WINDOW_AUTOSIZE);
  }
  std::ostringstream log;
  bool is_first_frame = true;
  double total_bytes = 0;
  boost::posix_time::ptime start_time;
  boost::posix_time::ptime previous_time;
  StreamReader* reader = stream->Subscribe();

  // Loop until the stopper thread signals that we need to stop.
  while (!stopped) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    if (frame == nullptr) {
      continue;
    } else if (frame->IsStopFrame()) {
      // We still need to check for stop frames, even though the stopper thread
      // is watching for stop frames at the highest level.
      break;
    } else {
      if (is_first_frame) {
        // We start recording after the first frame in order to reduce
        // experimental error when calculating the network bandwidth usage.
        is_first_frame = false;
        start_time = boost::posix_time::microsec_clock::local_time();
        previous_time = start_time;
      } else {
        current_image = frame->GetValue<cv::Mat>("original_image");
        total_bytes += frame->GetRawSizeBytes(
            std::unordered_set<std::string>{fields.begin(), fields.end()});

        // Calculate the network bandwidth.
        boost::posix_time::ptime current_time =
            boost::posix_time::microsec_clock::local_time();
        // Use total_microseconds() to avoid dividing by zero if less than a
        // second has passed.
        double net_bw_bps = (total_bytes * 8 /
                             (current_time - start_time).total_microseconds()) *
                            1000000;

        double fps = reader->GetHistoricalFps();

        long latency_micros =
            (current_time - frame->GetValue<boost::posix_time::ptime>(
                                Camera::kCaptureTimeMicrosKey))
                .total_microseconds();

        // Assemble log message;
        std::ostringstream msg;
        if (frame->GetValue<std::vector<int>>("ImageMatch.matches").size() >=
            1) {
          if (display) {
            last_match = frame->GetValue<cv::Mat>("original_image");
          }
          net_bw_bps = 1;
        } else {
          net_bw_bps = 0;
        }

        long it_micros = frame
                             ->GetValue<boost::posix_time::time_duration>(
                                 "ImageTransformer.total_micros")
                             .total_microseconds();
        long nne_micros = frame
                              ->GetValue<boost::posix_time::time_duration>(
                                  "NeuralNetEvaluator.total_micros")
                              .total_microseconds();
        long fvg_micros = frame
                              ->GetValue<boost::posix_time::time_duration>(
                                  "FvGen.total_micros")
                              .total_microseconds();
        long im_micros = frame
                             ->GetValue<boost::posix_time::time_duration>(
                                 "ImageMatch.total_micros")
                             .total_microseconds();

        int physical_kb = 0;
        int virtual_kb = 0;
        if (log_memory) {
          physical_kb = GetPhysicalKB();
          virtual_kb = GetVirtualKB();
        }

        msg << net_bw_bps << "," << fps << "," << latency_micros << ","
            << it_micros << "," << nne_micros << "," << fvg_micros << ","
            << im_micros << "," << physical_kb << "," << virtual_kb;
        if (display) {
          cv::imshow("current_image", current_image);
          cv::imshow("last_match", last_match);
          cv::waitKey(1);
        }

        if ((current_time - previous_time).total_seconds() >= 1) {
          // Every two seconds, log a frame's metrics to the console so that the
          // user can verify that the program is making progress.
          previous_time = current_time;

          std::cout << msg.str() << std::endl;
          log << msg.str() << std::endl;
        }
      }
    }
  }
  reader->UnSubscribe();

  std::ostringstream log_filepath;
  log_filepath << output_dir << "/ff_" << idx << "_"
               << boost::posix_time::to_iso_extended_string(log_time) << ".csv";
  std::ofstream log_file(log_filepath.str());
  log_file << "# network bandwidth (bps),fps,e2e latency (micros),"
              "Transformer micros,NNE micros,FV crop micros,"
              "imagematch micros,physical kb,virtual kb,"
              "caffe setup,caffe inference,caffe blob"
           << std::endl
           << log.str();
  log_file.close();
}

volatile int frames_processed = 0;
volatile int last_match = 0;

void Slack(StreamPtr stream, const std::string& slack_url) {
  CURL* curl;
  CURLcode res;

  curl_global_init(CURL_GLOBAL_ALL);

  StreamReader* reader = stream->Subscribe();
  while (!stopped) {
    std::unique_ptr<Frame> frame = reader->PopFrame();
    frames_processed += 1;
    if (frame == nullptr) {
      continue;
    } else if (frame->IsStopFrame()) {
      // We still need to check for stop frames, even though the stopper thread
      // is watching for stop frames at the highest level.
      break;
    } else {
      if (frame->Count("ImageMatch.matches") &&
          frame->GetValue<std::vector<int>>("ImageMatch.matches").size()) {
        int skipped_frames = frames_processed - last_match;
        last_match = frames_processed;
        float match_prob = frame->GetValue<float>("imagematch.match_prob");
        std::string frame_path =
            frame->GetValue<std::string>(JpegWriter::kRelativePathKey);
        std::string frame_link =
            "http://istc-vcs.pc.cc.cmu.edu:8000/" + frame_path;

        std::string msg =
            "{\"text\":\"Skipped " + std::to_string(skipped_frames) +
            " Match confidence (" + std::to_string(match_prob) + "): <" +
            frame_link +
            "2|frame>\n"
            "Path to full image: <" +
            frame_link + "|" + frame_link.substr(7, frame_link.size() - 7) +
            ">\"\n}";
        curl = curl_easy_init();
        if (curl) {
          curl_easy_setopt(curl, CURLOPT_URL, slack_url.c_str());
          curl_easy_setopt(curl, CURLOPT_POSTFIELDS, msg.c_str());
          res = curl_easy_perform(curl);
          if (res != CURLE_OK) {
            LOG(INFO) << "Curl failed: " << curl_easy_strerror(res);
          }
        }
      }
    }
  }

  curl_global_cleanup();
  reader->UnSubscribe();
}

// Submit the provided queries to ImageMatch, and add the required layers to the
// NNE and crops to the FvGen.
void AddQueries(std::vector<query_spec_t> queries,
                std::shared_ptr<NeuralNetEvaluator> nne,
                std::shared_ptr<FvGen> fvgen, std::shared_ptr<ImageMatch> im) {
  for (const auto& query : queries) {
    for (int k = 0; k < query.num; ++k) {
      nne->PublishLayer(query.layer);
      fvgen->AddFv(query.layer, query.xmin, query.ymin, query.xmax, query.ymax,
                   query.flat);
      im->AddQuery(query.path, query.layer, query.threshold, query.xmin,
                   query.ymin, query.xmax, query.ymax, query.flat);
    }
  }
}

void Run(const std::string& ff_conf, unsigned int num_frames, bool block,
         size_t queue_size, bool use_camera, const std::string& camera_name,
         const std::string& publish_url, unsigned int file_fps,
         int throttled_fps, unsigned int tokens, const std::string& model,
         size_t nne_batch_size, std::vector<std::string> fields,
         const std::string& output_dir, bool save_matches, bool log_memory,
         bool display, bool slack, const std::string& slack_url, int rotate) {
  boost::posix_time::ptime log_time =
      boost::posix_time::microsec_clock::local_time();

  boost::filesystem::path output_dir_path(output_dir);
  boost::filesystem::create_directory(output_dir_path);

  // Parse the ff_conf file.
  bool on_first_line = true;
  std::vector<std::string> kd_fv_keys;
  std::vector<std::pair<float, size_t>> buf_params;
  std::unordered_map<int, std::vector<query_spec_t>> level_to_queries;
  std::ifstream ff_conf_file(ff_conf);
  std::string line;
  unsigned int level_counter = 0;
  while (std::getline(ff_conf_file, line)) {
    std::vector<std::string> args = SplitString(line, ",");
    if (StartsWith(line, "#")) {
      // Ignore comment lines.
      continue;
    }
    CHECK(args.size() >= 3)
        << "Each line of the ff_conf file must contain at least three items: "
        << "Selectivity,BufferLength,NumQueries";

    float sel = std::stof(args.at(0));
    unsigned long buf_len = std::stoul(args.at(1));
    std::string kd_fv_key = args.at(2);
    for (decltype(args.size()) i = 3; i < args.size();) {
      CHECK(i + 8 < args.size()) << "Malformed configuration file";
      query_spec_t cur_query_spec;
      cur_query_spec.num = std::atoi(args.at(i++).c_str());
      cur_query_spec.layer = args.at(i++);
      cur_query_spec.xmin = std::atoi(args.at(i++).c_str());
      cur_query_spec.ymin = std::atoi(args.at(i++).c_str());
      cur_query_spec.xmax = std::atoi(args.at(i++).c_str());
      cur_query_spec.ymax = std::atoi(args.at(i++).c_str());
      cur_query_spec.flat = args.at(i++) == "true";
      cur_query_spec.path = args.at(i++);
      cur_query_spec.threshold = std::stof(args.at(i++));
      level_to_queries[level_counter].push_back(cur_query_spec);
    }

    std::ostringstream msg;
    msg << "Level " << level_counter << " - ";
    if (on_first_line) {
      on_first_line = false;
    } else {
      kd_fv_keys.push_back(kd_fv_key);
      buf_params.push_back({sel, (size_t)buf_len});
      msg << "selectivity: " << sel << ", buffer length: " << buf_len << ", ";
    }
    LOG(INFO) << msg.str();
    ++level_counter;
  }

  std::vector<std::shared_ptr<Processor>> procs;
  std::vector<std::thread> logger_threads;

  StreamPtr input_stream;
  if (use_camera) {
    // Create Camera.
    std::shared_ptr<Camera> camera =
        CameraManager::GetInstance().GetCamera(camera_name);

    if (camera->GetCameraType() != CameraType::CAMERA_TYPE_GST) {
      throw(std::invalid_argument("Must use a GStreamer camera"));
    }
    std::shared_ptr<GSTCamera> gst_camera =
        std::dynamic_pointer_cast<GSTCamera>(camera);
    // Why is this false?
    gst_camera->SetBlockOnPush(false);
    gst_camera->SetOutputFilepath(output_dir + "/" + camera_name + ".mp4");
    gst_camera->SetFileFramerate(file_fps);
    procs.push_back(gst_camera);
    input_stream = gst_camera->GetStream();
  } else {
    // Create FrameSubscriber.
    auto subscriber = std::make_shared<FrameSubscriber>(publish_url);
    // This is false because the "use_camera" case is false.
    subscriber->SetBlockOnPush(false);
    procs.push_back(subscriber);
    input_stream = subscriber->GetSink();
  }

  StreamPtr correct_fps_stream = input_stream;
  if (throttled_fps > 0) {
    // If we are supposed to throttler the stream in software, then create a
    // Throttler.
    auto throttler = std::make_shared<Throttler>(throttled_fps);
    throttler->SetSource(input_stream);
    throttler->SetBlockOnPush(block);
    procs.push_back(throttler);
    correct_fps_stream = throttler->GetSink();
  }

  // Create a FlowControlEntrance.
  auto fc_entrance = std::make_shared<FlowControlEntrance>(tokens);
  fc_entrance->SetSource(correct_fps_stream);
  fc_entrance->SetBlockOnPush(block);
  procs.push_back(fc_entrance);

  // Create an ImageTransformer.
  ModelDesc model_desc = ModelManager::GetInstance().GetModelDesc(model);
  Shape input_shape(3, model_desc.GetInputWidth(), model_desc.GetInputHeight());
  auto transformer =
      std::make_shared<ImageTransformer>(input_shape, true, rotate);
  transformer->SetSource(fc_entrance->GetSink());
  transformer->SetBlockOnPush(block);
  procs.push_back(transformer);

  // Create a NeuralNetEvaluator.
  auto nne = std::make_shared<NeuralNetEvaluator>(model_desc, input_shape,
                                                  nne_batch_size);
  nne->SetSource(transformer->GetSink());
  nne->SetBlockOnPush(block);
  procs.push_back(nne);

  // Create an FvGen.
  auto fvgen = std::make_shared<FvGen>();
  fvgen->SetSource(nne->GetSink());
  fvgen->SetBlockOnPush(block);
  procs.push_back(fvgen);
  StreamPtr fvgen_stream = fvgen->GetSink();

  // Create ImageMatch level 0. Use the same batch size as the
  // NeuralNetEvaluator.
  auto im_0 = std::make_shared<ImageMatch>(nne_batch_size);
  im_0->SetSource(fvgen_stream);
  im_0->SetBlockOnPush(block);
  procs.push_back(im_0);

  // Add the level 0 queries.
  AddQueries(level_to_queries[0], nne, fvgen, im_0);

  // Create a FlowControlExit.
  auto fc_exit = std::make_shared<FlowControlExit>();
  fc_exit->SetSource(im_0->GetSink());
  fc_exit->SetBlockOnPush(block);
  procs.push_back(fc_exit);

  // Create a logger thread to calculate statistics about the first ImageMatch
  // level's output stream. This Logger also might display the frames.
  StreamPtr fc_exit_sink = fc_exit->GetSink();
  logger_threads.push_back(std::thread([fc_exit_sink, log_time, fields,
                                        output_dir, log_memory, display] {
    Logger(0, fc_exit_sink, log_time, fields, output_dir, log_memory, display);
  }));

  StreamPtr jpeg_stream = nullptr;
  if (save_matches) {
    // Create JpegWriter.
    auto jpeg_writer =
        std::make_shared<JpegWriter>(JPEG_WRITER_FIELD, output_dir, true);
    jpeg_writer->SetSource(fc_exit_sink);
    procs.push_back(jpeg_writer);
    jpeg_stream = jpeg_writer->GetSink();

    // Create FrameWriter.
    auto frame_writer = std::make_shared<FrameWriter>(
        FRAME_WRITER_FIELDS, output_dir, FrameWriter::FileFormat::JSON, false,
        true);
    frame_writer->SetSource(fc_exit_sink);
    procs.push_back(frame_writer);
  }

  // Create additional keyframe detector + ImageMatch levels in the hierarchy.
  StreamPtr kd_input_stream = fvgen_stream;
  int count = 0;
  for (auto& param : buf_params) {
    LOG(INFO) << "Creating KeyframeDetector level " << count++
              << " with parameters: " << param.first << " " << param.second;
  }
  for (decltype(level_to_queries.size()) i = 1; i < level_to_queries.size();
       ++i) {
    // Create a keyframe detector.
    std::string kd_fv_key = kd_fv_keys.at(i - 1);
    std::pair<float, size_t> kd_buf_params = buf_params.at(i - 1);
    std::vector<std::pair<float, size_t>> kd_buf_params_vec = {kd_buf_params};
    fvgen->AddFv(kd_fv_key);
    nne->PublishLayer(kd_fv_key);
    auto kd = std::make_shared<KeyframeDetector>(kd_fv_key, kd_buf_params_vec);
    kd->SetSource(kd_input_stream);
    kd->SetBlockOnPush(block);
    procs.push_back(kd);
    kd_input_stream = kd->GetSink(kd->GetSinkName(0));

    // Create an ImageMatch.
    unsigned int kd_batch_size =
        ceil(kd_buf_params.first * kd_buf_params.second);
    auto additional_im = std::make_shared<ImageMatch>(kd_batch_size);
    additional_im->SetSource(kd->GetSink(kd->GetSinkName(0)));
    additional_im->SetBlockOnPush(block);
    procs.push_back(additional_im);

    // Add the level i queries.
    AddQueries(level_to_queries[i], nne, fvgen, additional_im);

    // Create a logger thread to calculate statistics about ImageMatch's output
    // stream.
    StreamPtr additional_im_sink = additional_im->GetSink();
    logger_threads.push_back(std::thread(
        [i, additional_im_sink, log_time, fields, output_dir, log_memory] {
          Logger(i + 1, additional_im_sink, log_time, fields, output_dir,
                 log_memory);
        }));
  }

  // Launch stopper thread.
  std::thread stopper_thread(
      [fc_exit_sink, num_frames] { Stopper(fc_exit_sink, num_frames); });

  // Launch Slack thread
  std::thread slack_thread;
  if (slack) {
    slack_thread = std::thread(
        [jpeg_stream, slack_url] { Slack(jpeg_stream, slack_url); });
  }

  // Start the processors in reverse order.
  for (auto procs_it = procs.rbegin(); procs_it != procs.rend(); ++procs_it) {
    (*procs_it)->Start(queue_size);
  }

  if (num_frames > 0) {
    while (!stopped) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } else {
    std::cout << "Press \"Enter\" to stop." << std::endl;
    getchar();
    stopped = true;
  }

  // Stop the processors in forward order.
  for (const auto& proc : procs) {
    proc->Stop();
  }

  // Join all of our helper threads.
  for (auto& t : logger_threads) {
    t.join();
  }
  stopper_thread.join();
  if (slack_thread.joinable()) {
    slack_thread.join();
  }
}

int main(int argc, char* argv[]) {
  po::options_description desc("FilterForward");
  desc.add_options()("help,h", "Print the help message.");
  desc.add_options()("config-dir,C", po::value<std::string>(),
                     "The directory containing streamer's configuration "
                     "files.");
  desc.add_options()("ff-conf,f", po::value<std::string>(),
                     "The file containing the keyframe detector's "
                     "configuration.");
  desc.add_options()("num-frames", po::value<unsigned int>()->default_value(0),
                     "The number of frames to run before automatically "
                     "stopping.");
  desc.add_options()("block,b",
                     "Whether processors should block when pushing frames.");
  desc.add_options()("queue-size,q", po::value<size_t>()->default_value(16),
                     "The size of the queues between processors.");
  desc.add_options()("camera,c", po::value<std::string>(),
                     "The name of the camera to use. Overrides "
                     "\"--publish-url\".");
  desc.add_options()("publish-url,u", po::value<std::string>(),
                     "The URL (host:port) on which the frame stream is being "
                     "published.");
  desc.add_options()("file-fps", po::value<unsigned int>()->default_value(0),
                     "Rate at which to read a file source (no effect if not "
                     "file source).");
  desc.add_options()("throttled-fps,i", po::value<int>()->default_value(0),
                     "The FPS at which to throttle (in software) the camera "
                     "stream. 0 means no throttling.");
  desc.add_options()("tokens,t", po::value<unsigned int>()->default_value(5),
                     "The number of flow control tokens to issue.");
  desc.add_options()("model,m", po::value<std::string>()->required(),
                     "The name of the model to evaluate.");
  desc.add_options()("nne-batch-size,s", po::value<size_t>()->default_value(1),
                     "nne batch size");
  desc.add_options()("fields",
                     po::value<std::vector<std::string>>()
                         ->multitoken()
                         ->composing()
                         ->required(),
                     "The fields to send over the network when calculating "
                     "theoretical network bandwidth usage.");
  desc.add_options()("output-dir,o", po::value<std::string>()->required(),
                     "The directory in which to write output files.");
  desc.add_options()(
      "save-matches",
      "Save JPEGs of frames matched by the first level of the hierarchy.");
  desc.add_options()("memory-usage", "Log memory usage.");
  desc.add_options()("display,d", "Enable display.");
  desc.add_options()("slack", po::value<std::string>(),
                     "Enable Slack notifications for matched frames, and send "
                     "notifications to the provided hook url.");
  desc.add_options()("rotate,r", po::value<int>()->default_value(0),
                     "The angle to rotate frames; must be 0, 90, 180, or 270.");

  // Parse the command line arguments.
  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, desc), args);
    if (args.count("help")) {
      std::cout << desc << std::endl;
      return 1;
    }
    po::notify(args);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  // Set up GStreamer.
  gst_init(&argc, &argv);
  // Set up glog.
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  // Initialize the streamer context. This must be called before using streamer.
  Context::GetContext().Init();

  // Extract the command line arguments.
  if (args.count("config-dir")) {
    Context::GetContext().SetConfigDir(args["config-dir"].as<std::string>());
  }
  std::string ff_conf = args["ff-conf"].as<std::string>();
  unsigned int num_frames = args["num-frames"].as<unsigned int>();
  bool block = args.count("block");
  size_t queue_size = args["queue-size"].as<size_t>();
  std::string camera;
  bool use_camera = args.count("camera");
  if (use_camera) {
    camera = args["camera"].as<std::string>();
  }
  std::string publish_url;
  if (args.count("publish-url")) {
    publish_url = args["publish-url"].as<std::string>();
  } else if (!use_camera) {
    throw std::runtime_error(
        "Must specify either \"--camera\" or \"--publish-url\".");
  }
  unsigned int file_fps = args["file-fps"].as<unsigned int>();
  int throttled_fps = args["throttled-fps"].as<int>();
  unsigned int tokens = args["tokens"].as<unsigned int>();
  std::string model = args["model"].as<std::string>();
  size_t nne_batch_size = args["nne-batch-size"].as<size_t>();
  std::vector<std::string> fields =
      args["fields"].as<std::vector<std::string>>();
  std::string output_dir = args["output-dir"].as<std::string>();
  bool save_matches = args.count("save-matches");
  bool log_memory = args.count("memory-usage");
  bool display = args.count("display");
  bool slack = args.count("slack");
  int rotate = args.count("rotate") ? args["rotate"].as<int>() : 0;
  std::string slack_url;
  if (slack) {
    slack_url = args["slack"].as<std::string>();
  }
  Run(ff_conf, num_frames, block, queue_size, use_camera, camera, publish_url,
      file_fps, throttled_fps, tokens, model, nne_batch_size, fields,
      output_dir, save_matches, log_memory, display, slack, slack_url, rotate);
  return 0;
}
