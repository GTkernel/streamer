// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "pipeline/spl_parser.h"

TEST(SPL_PARSER_TEST, TEST_PARSE) {
  std::string pipeline_desc = R"pipeline(
classifier = processor(ImageClassifier, model=AlexNet)
ip_cam = camera(GST_TEST)
transformer = processor(ImageTransformer, height=227, width=227)

transformer[input] = ip_cam[output]
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
  EXPECT_EQ(statement1.processor_type, PROCESSOR_TYPE_IMAGE_CLASSIFIER);

  auto statement4 = statements[3];
  EXPECT_EQ(statement4.statement_type, SPL_STATEMENT_CONNECT);
  EXPECT_EQ(statement4.lhs_processor_name, "transformer");
  EXPECT_EQ(statement4.lhs_stream_name, "input");
  EXPECT_EQ(statement4.rhs_processor_name, "ip_cam");
  EXPECT_EQ(statement4.rhs_stream_name, "output");
}

TEST(SPL_PARSER_TEST, TEST_COMMENT) {
  std::string pipeline_desc = R"pipeline(
# classifier = processor(ImageClassifier, model=AlexNet)
ip_cam = camera(GST_TEST)
transformer = processor(ImageTransformer, height=227, width=227)

transformer[input] = ip_cam[output]
# classifier[input] = transformer[output]
)pipeline";

  SPLParser parser;
  std::vector<SPLStatement> statements;
  bool result = parser.Parse(pipeline_desc, statements);

  EXPECT_TRUE(result);

  EXPECT_EQ(statements.size(), 3);

  auto stmt = statements[1];
  EXPECT_EQ(SPL_STATEMENT_PROCESSOR, stmt.statement_type);
  EXPECT_EQ("transformer", stmt.processor_name);
  EXPECT_EQ(PROCESSOR_TYPE_IMAGE_TRANSFORMER, stmt.processor_type);
}
