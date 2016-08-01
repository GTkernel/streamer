#include "gst_video_capture.h"
#ifdef USE_GIE
  #include "GIEClassifier.h"
#else
  #include "CaffeClassifier.h"
#endif
#include <iomanip>

int
main (int argc, char *argv[])
{
  if (argc != 6 && argc != 7) {
    std::cout << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt RTSPURI"
              << " display" << std::endl;
    std::cout << "";
    std::cout << "  RTSPURI: uri to the camera. e.g rtsp://xxx" << std::endl;
    std::cout << "  display: enable display or not, must have a X window" << std::endl;
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
  bool display = false;
  if (argc == 7) {
    display = true;
  }

#ifdef USE_GIE
  GIEClassifier classifier(model_file, trained_file, mean_file, label_file);
#else
  CaffeClassifier<float16, CAFFE_FP16_MTYPE> classifier(model_file, trained_file, mean_file, label_file);
  // CaffeClassifier<float, float> classifier(model_file, trained_file, mean_file, label_file);
#endif

  GstVideoCapture cap;
  if (!cap.CreatePipeline(argv[5])) {
    LOG(FATAL) << "Can't create pipeline, check camera and pipeline uri";
    exit(1);
  }

  if (display) {
    cv::namedWindow("camera");
  }

  while(1) {
    cv::Mat frame = cap.TryGetFrame();

    if (!frame.empty()) {
      std::vector<Prediction> predictions = classifier.Classify(frame, 1);
      Prediction p = predictions[0];
      LOG(INFO) << std::fixed << std::setprecision(4) << p.second << " - \""
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
