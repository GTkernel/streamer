//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#ifndef STREAMER_PARSER_H
#define STREAMER_PARSER_H

#include "common/common.h"
#include "pipeline.h"

enum SPL_STATEMENT_TYPE {
  SPL_STATEMENT_INVALID,
  SPL_STATEMENT_PROCESSOR,
  SPL_STATEMENT_CONNECT
};

struct SPLStatement {
  SPL_STATEMENT_TYPE statement_type;
  string processor_name;
  string processor_type;
  string lhs_processor_name;
  string rhs_processor_name;
  string lhs_stream_name;
  string rhs_stream_name;

  std::unordered_map<string, string> params;
};

/**
 * @brief Streamer pipeline parser. This is a VERY VERY SIMPLE AND BRUTE FORCE
 * IMPLEMENTATION!!
 */
class SPLParser {
 public:
  SPLParser();
  bool Parse(const string &spl, std::vector<SPLStatement> &statements);

 private:
  bool ValidName(const string &str);
};

#endif  // STREAMER_PARSER_H
