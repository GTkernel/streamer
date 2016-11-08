//
// Created by Ran Xian (xranthoar@gmail.com) on 10/28/16.
//

/**
 * @brief A demo showing how to stream, control and record from a GigE camera.
 */

#include "file_writer.h"
#include "streamer.h"

#include <boost/program_options.hpp>
#include <csignal>

namespace po = boost::program_options;
using std::cout;
using std::endl;

/**
 * @brief Overlay text onto an image
 * @param img The image to add text to
 * @param text The text to be added.
 * @param nrow The row of the text in the image.
 */
void AddText(cv::Mat &img, const string &text, int nrow) {
  // Maximum 10 lines of text
  const int MAX_LINE = 13;
  const int FONT_FACE = CV_FONT_HERSHEY_SIMPLEX;
  const double FONT_SCALE = 0.6;
  const int THICKNESS = 1;
  const cv::Scalar TEXT_COLOR(255, 255, 255);
  const int START_X = 10;
  const int START_Y = 20;
  const int TEXT_HEIGHT = 25;

  int width = img.cols;
  int height = img.rows;

  CHECK(nrow < MAX_LINE);

  cv::Point text_point(START_X, START_Y + (int)(TEXT_HEIGHT * nrow));
  cv::putText(img, text, text_point, FONT_FACE, FONT_SCALE, TEXT_COLOR,
              THICKNESS, CV_AA);
}

/**
 * @brief Add a gray background to the left of the image to make text more
 * salient.
 * @param img The image to add text onto.
 */
void AddGrayBackground(cv::Mat &img) {
  int width = img.cols;
  int height = img.rows;
  cv::Mat roi = img(cv::Rect(0, 0, 300, height));
  cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
  double alpha = 0.9;
  cv::addWeighted(color, alpha, roi, 1 - alpha, 0.0, roi);
}

void StartUp() {
#ifdef USE_VIMBA
  auto res = AVT::VmbAPI::VimbaSystem::GetInstance().Startup();
  CHECK(res == VmbErrorSuccess) << "Can't start up Vimba system";
#endif
}

void CleanUp() {
#ifdef USE_VIMBA
  auto res = AVT::VmbAPI::VimbaSystem::GetInstance().Shutdown();
  CHECK(res == VmbErrorSuccess) << "Can't shut down Vimba system";
#endif
}

void Run(const string &camera_name, const string &output_filename, bool display,
         int frames_per_file) {
  StartUp();

  auto &camera_manager = CameraManager::GetInstance();
  auto camera = camera_manager.GetCamera(camera_name);

  CHECK(camera->GetCameraType() == CAMERA_TYPE_PTGRAY ||
        camera->GetCameraType() == CAMERA_TYPE_VIMBA)
      << "Not running with GigE camera, we support PtGray and AlliedVision "
         "camera now";

  auto camera_reader = camera->GetSink("bgr_output")->Subscribe();

  auto bytes_stream = camera->GetSink("raw_output");
  std::shared_ptr<FileWriter> file_writer(new FileWriter("", frames_per_file));
  file_writer->SetSource("input", bytes_stream);

  camera->Start();
  STREAMER_SLEEP(10);

  if (display) {
    cv::namedWindow("Camera");
  }
  while (true) {
    cv::Mat image = camera_reader->PopFrame<ImageFrame>()->GetOriginalImage();
    cv::Mat image_to_show;
    if (display) {
      int width = image.cols;
      int height = image.rows;
      int new_width = 1280;
      int new_height = (int)((double)new_width / width * height);
      cv::resize(image, image_to_show, cv::Size(new_width, new_height));

      AddGrayBackground(image_to_show);
      //// Overlay text of camera information
      int row_idx = 0;
      AddText(image_to_show, "Parameters:", row_idx++);
      AddText(image_to_show,
              string() + "[R] Record: " +
                  (file_writer->IsStarted() ? file_writer->GetCurrentFilename()
                                            : "NO"),
              row_idx++);
      AddText(image_to_show, string() + "[E] Exposure: " +
                                 std::to_string(camera->GetExposure()),
              row_idx++);
      AddText(image_to_show, string() + "[S] Sharpness: " +
                                 std::to_string(camera->GetSharpness()),
              row_idx++);
      AddText(image_to_show,
              string() + "[H] Img Size: " +
                  std::to_string(camera->GetImageSize().width) + "x" +
                  std::to_string(camera->GetImageSize().height),
              row_idx++);
      AddText(image_to_show,
              string() + "[Z] Shutter: " +
                  std::to_string(camera->GetShutterSpeed()) + "ms",
              row_idx++);
      AddText(image_to_show, string() + "[V] Hue: " +
                                 std::to_string(camera->GetHue()) + " deg",
              row_idx++);
      AddText(image_to_show, string() + "[U] Saturation: " +
                                 std::to_string(camera->GetSaturation()) + "%",
              row_idx++);
      AddText(image_to_show, string() + "[B] Brightness: " +
                                 std::to_string(camera->GetBrightness()) + "%",
              row_idx++);
      AddText(image_to_show,
              string() + "[G] Gamma: " + std::to_string(camera->GetGamma()),
              row_idx++);
      AddText(image_to_show, string() + "[N] Gain: " +
                                 std::to_string(camera->GetGain()) + "dB",
              row_idx++);
      AddText(image_to_show,
              string() + "[O,P] WB " + "R:" +
                  std::to_string((int)camera->GetWBRed()) + " B:" +
                  std::to_string((int)camera->GetWBBlue()),
              row_idx++);
      AddText(
          image_to_show,
          string() + "[M] Color: " +
              (camera->GetPixelFormat() != CAMERA_PIXEL_FORMAT_MONO8 ? "YES"
                                                                     : "MONO"),
          row_idx++);

      cv::imshow("Camera", image_to_show);

      //// Keyboard controls
      char k = (char)cv::waitKey(15);
      if (k == 'q') {
        break;
      } else if (k == 'e') {
        camera->SetExposure(camera->GetExposure() * 0.95f);
      } else if (k == 'E') {
        camera->SetExposure(camera->GetExposure() * 1.05f);
      } else if (k == 's') {
        camera->SetSharpness(camera->GetSharpness() * 0.95f);
      } else if (k == 'S') {
        cout << "Increase sharpness" << endl;
        camera->SetSharpness(camera->GetSharpness() * 1.05f + 0.5f);
      } else if (k == 'H') {
        camera->SetImageSizeAndMode(Shape(1600, 1200), CAMERA_MODE_0);
      } else if (k == 'h') {
        camera->SetImageSizeAndMode(Shape(800, 600), CAMERA_MODE_2);
      } else if (k == 'b') {
        camera->SetBrightness(camera->GetBrightness() * 0.95f);
      } else if (k == 'B') {
        camera->SetBrightness(camera->GetBrightness() * 1.05f + 0.5f);
      } else if (k == 'z') {
        camera->SetShutterSpeed(camera->GetShutterSpeed() * 0.95f);
      } else if (k == 'Z') {
        camera->SetShutterSpeed(camera->GetShutterSpeed() * 1.05f);
      } else if (k == 'u') {
        camera->SetSaturation(camera->GetSaturation() * 0.95f);
      } else if (k == 'U') {
        camera->SetSaturation(camera->GetSaturation() * 1.05f + 0.5f);
      } else if (k == 'v') {
        camera->SetHue(camera->GetHue() * 0.95f);
      } else if (k == 'V') {
        camera->SetHue(camera->GetHue() * 1.05f + 0.5f);
      } else if (k == 'g') {
        camera->SetGamma(camera->GetGamma() * 0.95f);
      } else if (k == 'G') {
        camera->SetGamma(camera->GetGamma() * 1.05f + 0.5f);
      } else if (k == 'n') {
        camera->SetGain(camera->GetGain() * 0.95f);
      } else if (k == 'N') {
        camera->SetGain(camera->GetGain() * 1.05f + 0.5f);
      } else if (k == 'o') {
        camera->SetWBRed(camera->GetWBRed() * 0.95f);
      } else if (k == 'O') {
        camera->SetWBRed(camera->GetWBRed() + 1 * 1.05f + 1);
      } else if (k == 'p') {
        camera->SetWBBlue(camera->GetWBBlue() * 0.95f);
      } else if (k == 'P') {
        camera->SetWBBlue(camera->GetWBBlue() * 1.05f + 1);
      } else if (k == 'm') {
        camera->SetPixelFormat(CAMERA_PIXEL_FORMAT_MONO8);
      } else if (k == 'M') {
        camera->SetPixelFormat(CAMERA_PIXEL_FORMAT_RAW12);
      } else if (k == 'R') {
        file_writer->Start();
      } else if (k == 'X') {
        camera->SetImageSizeAndMode(Shape(2448, 2048), CAMERA_MODE_0);
      } else if (k == 'r') {
        if (file_writer->IsStarted()) file_writer->Stop();
      }
    }
  }

  camera_reader->UnSubscribe();
  camera->Stop();
  if (file_writer->IsStarted()) file_writer->Stop();

  CleanUp();
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
  //  desc.add_options()(
  //      "size,s",
  //      po::value<string>()->value_name("SIZE")->default_value("800x600"),
  //      "The size of image");
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
  Run(camera_name, output_filename, display, 1);

  return 0;
}
