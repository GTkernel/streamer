#define CPU_ONLY

#include "GstVideoCapture.h"
#include "CaffeClassifier.h"

int
main (int argc, char *argv[])
{
  if (argc != 6) {
    std::cout << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt RSTPURI"<< std::endl;
    exit(1);
  }

  // glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  CaffeClassifier classifier(model_file, trained_file, mean_file, label_file);

  GstVideoCapture cap;
  if (!cap.CreatePipeline(argv[5])) {
    LOG(FATAL) << "Can't create pipeline, check camera and pipeline uri";
    exit(1);
  }

  cv::namedWindow("camera");
  while(1) {
    cv::Mat frame = cap.GetFrame();

    if (!frame.empty()) {
      std::vector<CaffeClassifier::Prediction> predictions = classifier.Classify(frame, 1);
      CaffeClassifier::Prediction p = predictions[0];
      LOG(INFO) << std::fixed << std::setprecision(4) << p.second << " - \""
                << p.first << "\"" << std::endl;
      cv::imshow("camera", frame);
      cv::waitKey(30);
    } else {
      LOG(INFO) << "Got empty frame";
      break;
    }
  }

  cap.DestroyPipeline();

  return 0;
}
