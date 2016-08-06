#include "gst_video_capture.h"

#ifdef USE_GIE
  #include "gie_classifier.h"
#else
  #include "caffe_v1_classifier.h"
// Comment this if using fp16
//  #include "caffe_classifier.h"
  #include "mxnet_classifier.h"
#endif

#include "mxnet_classifier.h"

inline bool ends_with(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

int
main (int argc, char *argv[])
{
  if (argc != 8) {
    std::cout << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt RTSPURI"
              << " DISPLAY MODEL_TYPE" << std::endl;
    std::cout << "";
    std::cout << "  RTSPURI: uri to the camera or path to image. e.g rtsp://xxx, cat.jpeg" << std::endl;
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
  if (model_type == "gie") {
    classifier.reset(new GIEClassifier(model_file, trained_file, mean_file, label_file));
  }
#else
  CHECK(model_type != "gie") << "Binary is not compiled with GIE enabled, recompile with -DGIE=true";

  if (model_type == "caffe") {
    classifier.reset(new CaffeV1Classifier<float>(model_file, trained_file, mean_file, label_file));
//    CaffeClassifier<float16, CAFFE_FP16_MTYPE> classifier(model_file, trained_file, mean_file, label_file);
  }
#endif
  if (model_type == "mxnet") {
    classifier.reset(new MXNetClassifier(model_file, trained_file, mean_file, label_file, 224, 224));
  }

  bool display = display_on == "true";
  if (display) {
    cv::namedWindow("camera");
  }

  if (ends_with(video_uri, "jpeg") || ends_with(video_uri, "png") || ends_with(video_uri, "jpg")) {
    cv::Mat image = cv::imread(video_uri, CV_LOAD_IMAGE_ANYCOLOR);
    CHECK(image.data != nullptr) <<  "Can't read image " << video_uri;
    if (display) {
      cv::namedWindow("image");
      cv::imshow("image", image);
    }

    std::vector<Prediction> predictions = classifier->Classify(image, 5);
    for (int i = 0; i < 5; i++) {
      Prediction p = predictions[i];
      LOG(INFO) << "Rank " << i << ": " << p.second << " - \""  << p.first << "\"";
    }

    cv::destroyAllWindows();

  } else {
    GstVideoCapture cap;
    if (!cap.CreatePipeline(video_uri)) {
      LOG(FATAL) << "Can't create pipeline, check camera and pipeline uri";
      exit(1);
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
          cv::waitKey(10);
        }
      }
      if (!cap.IsConnected()) {
        LOG(INFO) << "Video capture lost connection";
        break;
      }
    }

    cap.DestroyPipeline();
  }

  return 0;
}
