#include "GstVideoCapture.h"

int
main (int argc, char *argv[])
{
  if (argc != 2) {
    std::cout << "Usage: ./tx1_run_alexnet RSTPURI" << std::endl;
    exit(1);
  }

  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);

  GstVideoCapture cap;
  if (!cap.CreatePipeline(argv[1])) {
    LOG(FATAL) << "Can't create pipeline, check camera and pipeline uri";
    exit(1);
  }

  cv::namedWindow("camera");
  while(1) {
    cv::Mat frame = cap.GetFrame();
    if (!frame.empty()) {
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