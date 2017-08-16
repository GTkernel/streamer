
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
#include "processor/object_detector.h"
#include "processor/caffe_mtcnn.h"
#include "processor/db_writer.h"
#include "processor/facenet.h"
#include "processor/obj_tracker.h"
#include "processor/object_tracker.h"
#include "processor/opencv_motion_detector.h"
#include "processor/opencv_people_detector.h"
#include "processor/yolo_detector.h"
#ifdef USE_SSD
#include "processor/ssd_detector.h"
#endif  // USE_SSD
#ifdef USE_NCS
#include "processor/ncs_yolo_detector.h"
#endif  // USE_NCS
#include "processor/processor.h"
#include "processor/processor_factory.h"
#include "utils/utils.h"
#include "video/gst_video_encoder.h"

#endif  // STREAMER_STREAMER_H_
