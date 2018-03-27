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

#ifndef STREAMER_PIPELINE_SPL_PARSER_H_
#define STREAMER_PIPELINE_SPL_PARSER_H_

#include "common/types.h"

enum SPLStatementType {
  SPL_STATEMENT_INVALID,
  SPL_STATEMENT_PROCESSOR,
  SPL_STATEMENT_CONNECT,
};

struct SPLStatement {
  SPLStatementType statement_type;
  std::string processor_name;
  ProcessorType processor_type;
  std::string lhs_processor_name;
  std::string rhs_processor_name;
  std::string lhs_stream_name;
  std::string rhs_stream_name;

  FactoryParamsType params;
};

/**
 * @brief Streamer pipeline parser. This is a VERY VERY SIMPLE AND BRUTE FORCE
 * IMPLEMENTATION!!
 */
class SPLParser {
 public:
  SPLParser();
  bool Parse(const std::string& spl, std::vector<SPLStatement>& statements);

 private:
  bool ValidName(const std::string& str);
};

#endif  // STREAMER_PIPELINE_SPL_PARSER_H_
