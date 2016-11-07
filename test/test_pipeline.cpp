//
// Created by Ran Xian (xranthoar@gmail.com) on 11/6/16.
//

#include "gtest/gtest.h"
#include "pipeline/pipeline.h"

TEST(PIPELINE_TEST, TEST_CONSTRUCT_PIPELINE) {
  string pipeline_desc = R"pipeline(
classifier = processor(ImageClassifier, model=AlexNet)
# ip_cam = camera(GST_TEST, width=1024, height=768)
# transformer = processor(ImageTransformer, height=227, width=227)

# transformer[input] = ip_cam[bgr_output]
# classifier[input] = transformer[output]
)pipeline";

  SPLParser parser;
  std::vector<SPLStatement> stmts;
  bool success = parser.Parse(pipeline_desc, stmts);
  EXPECT_TRUE(success);
  auto pipeline = Pipeline::ConstructPipeline(stmts);

  EXPECT_TRUE(pipeline != nullptr);
  EXPECT_TRUE(pipeline->GetProcessor("classifier") != nullptr);
}