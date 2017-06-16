#include "camera/camera_manager.h"
#include "common/common.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "pipeline/pipeline.h"
#include "pipeline/spl_parser.h"
#include "processor/dummy_nn_processor.h"
#include "processor/image_classifier.h"
#ifdef USE_FRCNN
#include "processor/object_detector.h"
#endif
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/opencv_face_detector.h"
#include "processor/opencv_people_detector.h"
#include "processor/object_tracker.h"
#include "processor/caffe_mtcnn.h"
#include "processor/facenet.h"
#include "processor/opencv_motion_detector.h"
#include "processor/struck_tracker.h"
#include "processor/db_writer.h"
#include "processor/ssd_detector.h"
#include "processor/processor.h"
#include "processor/processor_factory.h"
#include "utils/utils.h"
#include "video/gst_video_encoder.h"
