#include "object_detector.h"
#include "model/model_manager.h"

#define GET_SOURCE_NAME(i) ("input" + std::to_string(i))
#define GET_SINK_NAME(i) ("output" + std::to_string(i))

ObjectDetector::ObjectDetector(const ModelDesc &model_desc,
                               Shape input_shape,
                               size_t batch_size)
        : Processor({}, {}),
          model_desc_(model_desc_),
          input_shape_(input_shape),
          batch_size_(batch_size) {
    for (size_t i = 0; i < batch_size_; i++) {
        sources_.insert({"input" + std::to_string(i), nullptr});
        sinks_.insert({"output" + std::to_string(i),
                       std::shared_ptr<Stream>(new Stream)});
    }

    LOG(INFO) << "batch size of " << batch_size_;
}

bool ObjectDetector::Init() {
    CHECK(model_desc_.GetLabelFilePath() != "") << "Model "
            << model_desc_.GetName() << " has an empty label file";
    std::ifstream labels(model_desc_.GetLabelFilePath());
    CHECK(labels) << "Unable to open labels file "
            << model_desc_.GetLabelFilePath();

    string line;
    while (std::getline(labels, line)) labels_.push_back(string(line));

    auto &manager = ModelManager::GetInstance();
    model_ = manager.CreateModel(model_desc_, input_shape_, batch_size_);
    model_->Load();

    auto mean_colors = manager.GetMeanColors();
    mean_image_ = cv::Mat(cv::Size(input_shape_.width, input_shape_.height),
            CV_32FC3,
            cv::Scalar(mean_colors[0], mean_colors[1], mean_colors[2]));

    input_buffer_ =
        DataBuffer(batch_size_ * input_shape_.GetSize() * sizeof(float));

    LOG(INFO) << "Classifier initialized";
    return true;
}

bool ObjectDetector::OnStop() {
    model_.reset(nullptr);
    return true;
}

void ObjectDetector::Process() {
    Timer timer;
    timer.Start();

    std::vector<std::shared_ptr<ImageFrame>> image_frames;
    float *data = (float *) input_buffer_.GetBuffer();
    for (int i = 0; i < batch_size_; i++) {
        auto image_frame = GetFrame<ImageFrame>(GET_SOURCE_NAME(i));
        image_frames.push_back(image_frame);
        cv::Mat img = image_frame->GetImage();
        CHECK(img.channels() == input_shape_.channel &&
                img.size[0] == input_shape_.width &&
                img.size[1] == input_shape_.height);
        std::vector<cv::Mat> output_channels;
        for (int j = 0; j < input_shape_.channel; j++) {
            cv::Mat channel(input_shape_.height, input_shape_.width,
                            CV_32FC1, data);
            output_channels.push_back(channel);
            data += input_shape_.width * input_shape_.height;
        }
        cv::split(img, output_channels);
    }

    LOG(INFO) << "Object detection took " << timer.ElapsedMSec() << " ms";
}

ProcessorType ObjectDetector::GetType() {
    return PROCESSOR_TYPE_OBJECT_DETECTOR;
}

void ObjectDetector::SetInputStream(int src_id, StreamPtr stream) {
    SetSource(GET_SOURCE_NAME(src_id), stream);
}
