//
// Created by Ran Xian (xranthoar@gmail.com) on 10/28/16.
//

/**
 * @brief A demo showing how to stream, control and record from a GigE camera.
 */

#include "streamer.h"

#include <camera/pgr_camera.h>
#include <boost/program_options.hpp>
#include <csignal>

namespace po = boost::program_options;
using std::cout;
using std::endl;

void Run(const string &camera_name, const string &output_filename,
         bool display) {
  auto &camera_manager = CameraManager::GetInstance();
  auto camera = camera_manager.GetCamera(camera_name);

  CHECK(camera->GetType() == CAMERA_TYPE_PTGRAY)
      << "Not running with GigE camera";

  auto ptgray_camera = dynamic_cast<PGRCamera *>(camera.get());
  ptgray_camera->Start();
  STREAMER_SLEEP(10);

  if (display) {
    cv::namedWindow("Image");
  }
  while (true) {
    cv::Mat image =
        ptgray_camera->GetSinks()[0]->PopFrame<ImageFrame>()->GetOriginalImage();
    if (display) {
      cv::imshow("Image", image);
      int k = cv::waitKey(10);
      if (k == 'q') {
        break;
      } else if (k == 'e') {
        ptgray_camera->SetExposure(ptgray_camera->GetExposure() * 0.95f);
      } else if (k == 'r') {
        ptgray_camera->SetExposure(ptgray_camera->GetExposure() * 1.05f);
      } else if (k == 's') {
        ptgray_camera->SetSharpness(ptgray_camera->GetSharpness() * 0.95f);
      } else if (k == 'd') {
        ptgray_camera->SetSharpness(ptgray_camera->GetSharpness() * 1.05f);
      } else if (k == 'h') {
        ptgray_camera->SetImageSizeAndVideoMode(Shape(1600, 1200),
                                                FlyCapture2::MODE_0);
      } else if (k == 'j') {
        ptgray_camera->SetImageSizeAndVideoMode(Shape(640, 480),
                                                FlyCapture2::MODE_2);
      }
    }
  }

  ptgray_camera->Stop();
}

int main(int argc, char *argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("GigE camera demo");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("camera",
                     po::value<string>()->value_name("CAMERA")->required(),
                     "The name of the camera to use");
  desc.add_options()("display,d", "Enable display or not");
  desc.add_options()(
      "output,o",
      po::value<string>()->value_name("OUTPUT")->default_value("camera.raw"),
      "The name of the file to store the raw video bytes");
  desc.add_options()("config_dir,C",
                     po::value<string>()->value_name("CONFIG_DIR"),
                     "The directory to find streamer's configurations");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << e.what() << endl;
    cout << desc << endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  ///////// Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<string>());
  }
  // Init streamer context, this must be called before using streamer.
  Context::GetContext().Init();

  auto camera_name = vm["camera"].as<string>();
  auto output_filename = vm["output"].as<string>();
  bool display = vm.count("display") != 0;
  Run(camera_name, output_filename, display);

  return 0;
}
