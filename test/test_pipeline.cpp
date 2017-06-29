//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include <streamer.h>
#include "gtest/gtest.h"
#include "pipeline/pipeline.h"

TEST(PIPELINE_TEST, TEST_CONSTRUCT_PIPELINE) {
  string pipeline_desc = R"pipeline(
# classifier = processor(ImageClassifier, model=AlexNet)
ip_cam = camera(GST_TEST)
transformer = processor(ImageTransformer, height=227, width=227)

transformer[input] = ip_cam[output]
# classifier[input] = transformer[output]
)pipeline";

  SPLParser parser;
  std::vector<SPLStatement> stmts;
  bool success = parser.Parse(pipeline_desc, stmts);
  EXPECT_TRUE(success);

  auto pipeline = Pipeline::ConstructPipeline(stmts);

  EXPECT_TRUE(pipeline != nullptr);

  auto ip_cam = pipeline->GetProcessor("ip_cam");
  auto transformer = pipeline->GetProcessor("transformer");

  EXPECT_TRUE(ip_cam != nullptr);
  EXPECT_TRUE(transformer != nullptr);

  // Start the pipeline
  success = pipeline->Start();
  EXPECT_TRUE(success);

  EXPECT_TRUE(ip_cam->IsStarted());
  EXPECT_TRUE(transformer->IsStarted());

  if (ip_cam->IsStarted() && transformer->IsStarted()) {
    auto output = transformer->GetSink("output");
    auto reader = output->Subscribe();
    auto frame = reader->PopFrame();
    const auto& transformed_image = frame->GetValue<cv::Mat>("image");
    const auto& original_image = frame->GetValue<cv::Mat>("original_image");
    reader->UnSubscribe();

    EXPECT_EQ(227, transformed_image.rows);
    EXPECT_EQ(227, transformed_image.cols);

    auto camera = CameraManager::GetInstance().GetCamera("GST_TEST");
    EXPECT_EQ(original_image.cols, camera->GetWidth());
    EXPECT_EQ(original_image.rows, camera->GetHeight());
  }

  // Stop the pipeline
  pipeline->Stop();
  EXPECT_TRUE(!ip_cam->IsStarted());
  EXPECT_TRUE(!transformer->IsStarted());
}
