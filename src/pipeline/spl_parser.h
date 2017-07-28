//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#ifndef STREAMER_PIPELINE_SPL_PARSER_H_
#define STREAMER_PIPELINE_SPL_PARSER_H_

#include "common/common.h"

enum SPLStatementType {
  SPL_STATEMENT_INVALID,
  SPL_STATEMENT_PROCESSOR,
  SPL_STATEMENT_CONNECT,
};

struct SPLStatement {
  SPLStatementType statement_type;
  string processor_name;
  ProcessorType processor_type;
  string lhs_processor_name;
  string rhs_processor_name;
  string lhs_stream_name;
  string rhs_stream_name;

  FactoryParamsType params;
};

/**
 * @brief Streamer pipeline parser. This is a VERY VERY SIMPLE AND BRUTE FORCE
 * IMPLEMENTATION!!
 */
class SPLParser {
 public:
  SPLParser();
  bool Parse(const string& spl, std::vector<SPLStatement>& statements);

 private:
  bool ValidName(const string& str);
};

#endif  // STREAMER_PIPELINE_SPL_PARSER_H_
