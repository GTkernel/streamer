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

#ifndef STREAMER_STREAMER_H_
#define STREAMER_STREAMER_H_

#include "camera/camera_manager.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "pipeline/pipeline.h"
#include "pipeline/spl_parser.h"
#ifdef USE_CAFFE
#include "processor/caffe_facenet.h"
#endif  // USE_CAFFE
#include "processor/db_writer.h"
#include "processor/detectors/object_detector.h"
#include "processor/detectors/opencv_people_detector.h"
#include "processor/face_tracker.h"
#include "processor/image_classifier.h"
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/opencv_motion_detector.h"
#include "processor/processor.h"
#include "processor/processor_factory.h"
#include "processor/trackers/object_tracker.h"
#ifdef USE_RPC
#include "processor/rpc/frame_receiver.h"
#include "processor/rpc/frame_sender.h"
#endif  // USE_RPC
#include "utils/utils.h"
#include "video/gst_video_encoder.h"

#endif  // STREAMER_STREAMER_H_
