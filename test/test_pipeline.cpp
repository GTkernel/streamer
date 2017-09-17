#include <string>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "camera/camera.h"
#include "camera/camera_manager.h"
#include "pipeline/pipeline.h"
#include "pipeline/spl_parser.h"
#include "processor/processor.h"
#include "stream/frame.h"

TEST(TestPipeline, TestConstructPipeline) {
  std::string pipeline_desc = R"pipeline(
# classifier = processor(ImageClassifier, model=AlexNet)
ip_cam = camera(GST_TEST)
transformer = processor(ImageTransformer, height=227, width=227)

transformer[input] = ip_cam[output]
# classifier[input] = transformer[output]
)pipeline";

  SPLParser parser;
  std::vector<SPLStatement> stmts;
  ASSERT_TRUE(parser.Parse(pipeline_desc, stmts));

  auto pipeline = Pipeline::ConstructPipeline(stmts);
  ASSERT_TRUE(pipeline != nullptr);
  auto ip_cam = pipeline->GetProcessor("ip_cam");
  ASSERT_TRUE(ip_cam != nullptr);
  auto transformer = pipeline->GetProcessor("transformer");
  ASSERT_TRUE(transformer != nullptr);

  // Start the pipeline
  ASSERT_TRUE(pipeline->Start());
  ASSERT_TRUE(ip_cam->IsStarted());
  ASSERT_TRUE(transformer->IsStarted());

  auto reader = transformer->GetSink("output")->Subscribe();
  auto frame = reader->PopFrame();
  const auto& transformed_image = frame->GetValue<cv::Mat>("image");
  const auto& original_image = frame->GetValue<cv::Mat>("original_image");

  ASSERT_EQ(227, transformed_image.rows);
  ASSERT_EQ(227, transformed_image.cols);

  auto camera = CameraManager::GetInstance().GetCamera("GST_TEST");
  ASSERT_EQ(original_image.cols, camera->GetWidth());
  ASSERT_EQ(original_image.rows, camera->GetHeight());

  // Stop the pipeline
  reader->UnSubscribe();
  pipeline->Stop();
  ASSERT_TRUE(!ip_cam->IsStarted());
  ASSERT_TRUE(!transformer->IsStarted());
}
