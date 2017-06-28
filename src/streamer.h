
#ifndef STREAMER_STREAMER_H_
#define STREAMER_STREAMER_H_

#include "camera/camera_manager.h"
#include "common/common.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "pipeline/pipeline.h"
#include "pipeline/spl_parser.h"
#ifdef USE_RPC
#include "processor/rpc/frame_receiver.h"
#include "processor/rpc/frame_sender.h"
#endif  // USE_RPC
#include "processor/image_classifier.h"
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/opencv_face_detector.h"
#include "processor/processor.h"
#include "processor/processor_factory.h"
#include "utils/utils.h"
#include "video/gst_video_encoder.h"

#endif  // STREAMER_STREAMER_H_
