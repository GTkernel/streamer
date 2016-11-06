//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#include "gtest/gtest.h"
#include "pipeline/spl_parser.h"

TEST(SPL_PARSER_TEST, TEST_PARSE) {
  string pipeline_desc = R"pipeline(
classifier = processor(ImageClassifier, model=AlexNet)
ip_cam = camera(AMC, width=1024, height=768)
transformer = processor(ImageTransformer, height=227, width=227)

transformer[input] = ip_cam[bgr_output]
classifier[input] = transformer[output]
)pipeline";

  SPLParser parser;
  std::vector<SPLStatement> statements;
  bool result = parser.Parse(pipeline_desc, statements);

  EXPECT_TRUE(result);

  EXPECT_EQ(statements.size(), 5);

  auto statement1 = statements[0];

  EXPECT_EQ(statement1.statement_type, SPL_STATEMENT_PROCESSOR);
  EXPECT_EQ(statement1.processor_name, "classifier");
  EXPECT_EQ(statement1.params["model"], "AlexNet");
  EXPECT_EQ(statement1.processor_type, "ImageClassifier");

  auto statement4 = statements[3];
  EXPECT_EQ(statement4.statement_type, SPL_STATEMENT_CONNECT);
  EXPECT_EQ(statement4.lhs_processor_name, "transformer");
  EXPECT_EQ(statement4.lhs_stream_name, "input");
  EXPECT_EQ(statement4.rhs_processor_name, "ip_cam");
  EXPECT_EQ(statement4.rhs_stream_name, "bgr_output");
}