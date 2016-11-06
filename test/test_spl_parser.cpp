//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#include "gtest/gtest.h"
#include "pipeline/spl_parser.h"

TEST(SPL_PARSER_TEST, TEST_PARSE) {
  string pipeline_desc = R"pipeline(
classifier  = processor(ImageClassifier, model=AlexNet)
)pipeline";

  SPLParser parser;
  Pipeline pipeline = parser.Parse(pipeline_desc);

  EXPECT_EQ(pipeline.GetProcessor("classifier")->GetType(),
            PROCESSOR_TYPE_IMAGE_CLASSIFIER);
}