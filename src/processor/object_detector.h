#ifndef STREAMER_OBJECT_DETECTOR_H
#define STREAMER_OBJECT_DETECTOR_H

#include "common/common.h"
#include "model/model.h"
#include "processor.h"

class ObjectDetector : public Processor {
public:
    ObjectDetector(const ModelDesc &model_desc,
                   Shape input_shape,
                   size_t batch_size);
    virtual ProcessorType GetType() override;
    void SetInputStream(int src_id, StreamPtr stream);

protected:
    virtual bool Init() override;
    virtual bool OnStop() override;
    virtual void Process() override;

private:
    DataBuffer input_buffer_;
    std::unique_ptr<Model> model_;
    std::vector<string> labels_;
    // TODO: Add variables for holding bounding boxes
    ModelDesc model_desc_;
    Shape input_shape_;
    cv::Mat mean_image_;
    size_t batch_size_;

};

#endif // STREAMER_OBJECT_DETECTOR_H
