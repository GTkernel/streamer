#include "gst_video_capture.h"

#ifdef USE_GIE
  #include "gie_classifier.h"
#else
  #include "caffe_v1_classifier.h"
// Comment this if using fp16
//  #include "caffe_classifier.h"
  #include "mxnet_classifier.h"
#endif

int
main (int argc, char *argv[])
{
  if (argc != 8) {
    std::cout << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt RTSPURI"
              << " DISPLAY MODEL_TYPE" << std::endl;
    std::cout << "";
    std::cout << "  RTSPURI: uri to the camera. e.g rtsp://xxx" << std::endl;
    std::cout << "  DISPLAY: true or false: enable display or not, must have a X window" << std::endl;
    std::cout << "  MODEL_TYPE: caffe, mxnet, gie";
    exit(1);
  }

  // Set up
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  // Get options
  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  string video_uri    = argv[5];
  string display_on   = argv[6];
  string model_type   = argv[7];

  // Check options
  CHECK(model_type == "caffe" || model_type == "gie" || model_type == "mxnet") << "MODEL_TYPE can only be one of caffe, mxnet or gie";

  std::unique_ptr<Classifier> classifier;
#ifdef USE_GIE
  CHECK(model_type != "caffe") << "Binary is compiled with GIE, can't run Caffe, recompile with -DGIE=false";
  classifier.reset(new GIEClassifier(model_file, trained_file, mean_file, label_file));
#else
  CHECK(model_type != "gie") << "Binary is not compiled with GIE enabled, recompile with -DGIE=true";

  if (model_type == "caffe") {
    classifier.reset(new CaffeV1Classifier<float>(model_file, trained_file, mean_file, label_file));
//    CaffeClassifier<float16, CAFFE_FP16_MTYPE> classifier(model_file, trained_file, mean_file, label_file);
  }
#endif
  if (model_type == "mxnet") {
//    classifier.reset(new MXNetClassifier(model_file, trained_file, mean_file, label_file, 224, 224));
  }

  GstVideoCapture cap;
  if (!cap.CreatePipeline(video_uri)) {
    LOG(FATAL) << "Can't create pipeline, check camera and pipeline uri";
    exit(1);
  }

  bool display = display_on == "true";
  if (display) {
    cv::namedWindow("camera");
  }

  while(1) {
    cv::Mat frame = cap.TryGetFrame();

    if (!frame.empty()) {
      std::vector<Prediction> predictions = classifier->Classify(frame, 1);
      Prediction p = predictions[0];
      LOG(INFO) << p.second << " - \""
                << p.first << "\"" << std::endl;
      if (display) {
      cv::imshow("camera", frame);
      cv::waitKey(30);
      }
    }
    if (!cap.IsConnected()) {
      LOG(INFO) << "Video capture lost connection";
      break;
    }
  }

  cap.DestroyPipeline();

  return 0;
}
