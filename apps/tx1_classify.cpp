//
// Created by Ran Xian on 7/28/16.
//

#include "GstVideoCapture.h"
//#include "CaffeClassifier.h"
#include "GIEClassifier.h"
#include <iomanip>

int
main (int argc, char *argv[])
{
  if (argc != 6 && argc != 7) {
    std::cout << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt IMAGE"
              << " display" << std::endl;
    std::cout << "";
    std::cout << "  IMAGE: image to be classified" << std::endl;
    std::cout << "  display: enable display or not, must have a X window" << std::endl;
    exit(1);
  }

  // Set up
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  // Get options
  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  string image_file   = argv[5];
  bool display = false;
  if (argc == 7) {
    display = true;
  }

//  CaffeFp16Classifier classifier(model_file, trained_file, mean_file, label_file);
  GIEClassifier classifier(model_file, trained_file, mean_file, label_file);
//  cap.SetTargetFrameSize(classifier.GetInputGeometry());

  cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);

  if (!image.data) {
    LOG(ERROR) << "Can't read image";
    return 1;
  }

  if (display) {
    cv::namedWindow("image");
    cv::imshow("image", image);
  }

  std::vector<GIEClassifier::Prediction> predictions = classifier.Classify(image, 5);
  for (int i = 0; i < 5; i++) {
    GIEClassifier::Prediction p = predictions[i];
    LOG(INFO) << "Rank " << i << ": " << p.second << " - \""  << p.first << "\"";
  }

  cv::destroyAllWindows();

  return 0;
}
