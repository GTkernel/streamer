
#include "throttler.h"
#include "model/model_manager.h"
#include "streamer.h"

Throttler::Throttler(int fps)
    : Processor(PROCESSOR_TYPE_THROTTLER, {"input"}, {"output"}), fps_(fps) {}

std::shared_ptr<Throttler> Throttler::Create(const FactoryParamsType& params) {
  int fps = std::stoi(params.at("fps"));
  return std::make_shared<Throttler>(fps);
}

bool Throttler::Init() {
  timer_.Start();
  SetFps(fps_);
  return true;
}

bool Throttler::OnStop() { return true; }

void Throttler::Process() {
  auto frame = GetFrame<Frame>("input");
  double elapsed_ms = timer_.ElapsedMSec();
  if (elapsed_ms < delay_ms_) {
    // Drop frame
    return;
  }
  // Restart timer
  timer_.Start();

  // Make new frame to pass through, conditioned on its type
  switch (frame->GetType()) {
    case FRAME_TYPE_LAYER: {
      auto layer_frame = std::dynamic_pointer_cast<LayerFrame>(frame);
      cv::Mat original_image = layer_frame->GetOriginalImage();
      cv::Mat activations = layer_frame->GetActivations();
      string layer_name = layer_frame->GetLayerName();
      PushFrame("output",
                new LayerFrame(layer_name, activations, original_image));
      break;
    }
    case FRAME_TYPE_IMAGE: {
      auto image_frame = std::dynamic_pointer_cast<ImageFrame>(frame);
      cv::Mat image = image_frame->GetImage();
      cv::Mat original_image = image_frame->GetOriginalImage();
      auto new_frame = new ImageFrame(image, original_image);
      PushFrame("output", new_frame);
      break;
    }
    case FRAME_TYPE_BYTES: {
      auto bytes_frame = std::dynamic_pointer_cast<BytesFrame>(frame);
      cv::Mat original_image = bytes_frame->GetOriginalImage();
      std::vector<char> data_buffer = bytes_frame->GetDataBuffer();
      PushFrame("output", new BytesFrame(data_buffer, original_image));
      break;
    }
    case FRAME_TYPE_MD: {
      auto metadata_frame = std::dynamic_pointer_cast<MetadataFrame>(frame);
      cv::Mat original_image = metadata_frame->GetOriginalImage();
      std::vector<string> tags = metadata_frame->GetTags();
      if (tags.size() == 0) {
        PushFrame("output", new MetadataFrame(tags, original_image));
      } else {
        std::vector<Rect> bboxes = metadata_frame->GetBboxes();
        PushFrame("output", new MetadataFrame(bboxes, original_image));
      }
      break;
    }
    default: { STREAMER_NOT_IMPLEMENTED; }
  }
}

void Throttler::SetFps(int fps) {
  if (fps == 0) {
    LOG(WARNING) << "Tried to set FPS to zero.";
    return;
  } else if (fps == -1) {
    // Turn throttling off with -1
    delay_ms_ = 0;
  } else {
    delay_ms_ = 1000 / fps;
  }
}
