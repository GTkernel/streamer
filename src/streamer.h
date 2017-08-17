
#ifndef STREAMER_STREAMER_H_
#define STREAMER_STREAMER_H_

#include "camera/camera_manager.h"
#include "common/common.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "pipeline/pipeline.h"
#include "pipeline/spl_parser.h"
#ifdef USE_CAFFE
#include "processor/caffe_facenet.h"
#endif  // USE_CAFFE
#include "processor/db_writer.h"
#include "processor/image_classifier.h"
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/detectors/object_detector.h"
#include "processor/trackers/object_tracker.h"
#include "processor/opencv_motion_detector.h"
#include "processor/face_tracker.h"
#include "processor/detectors/opencv_people_detector.h"
#include "processor/processor.h"
#include "processor/processor_factory.h"
#ifdef USE_RPC
#include "processor/rpc/frame_receiver.h"
#include "processor/rpc/frame_sender.h"
#endif  // USE_RPC
#include "utils/utils.h"
#include "video/gst_video_encoder.h"

#endif  // STREAMER_STREAMER_H_
