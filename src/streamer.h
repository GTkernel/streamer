#include "camera/camera_manager.h"
#include "common/common.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "processor/dummy_nn_processor.h"
#include "processor/image_classifier.h"
#include "processor/image_classifier.h"
#include "processor/image_segmenter.h"
#include "processor/image_transformer.h"
#include "processor/opencv_face_detector.h"
#include "processor/processor.h"
#include "utils/utils.h"
#include "video/gst_video_encoder.h"