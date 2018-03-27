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

#include "spl_parser.h"

SPLParser::SPLParser() {}

// TODO: this is definitely going to be rewrite..
class Tokenizer {
 public:
  Tokenizer(const std::string& str) : str_(str), index_(0) {}

  bool HasNext() { return index_ != str_.length(); }

  std::string NextToken() {
    size_t end = str_.length();
    if (index_ == end) return "";

    size_t start = 0;
    bool token_started = false;
    while (index_ != end) {
      char ch = str_.data()[index_];

      if (ch == '#') {
        while (index_ != end) {
          if (str_.data()[index_] == '\n') {
            break;
          }
          index_ += 1;
        }
        continue;
      }

      // Return special character
      if (ch == '=' || ch == '(' || ch == ')' || ch == '[' || ch == ']' ||
          ch == ',' || ch == '\n') {
        if (!token_started) {
          index_ += 1;
          return std::string(1, ch);
        } else {
          return str_.substr(start, index_ - start);
        }
      }

      if (ch == ' ' || ch == '\t') {
        if (token_started) {
          // Token has started
          return str_.substr(start, index_ - start);
        }
      } else {
        if (!token_started) {
          start = index_;
          token_started = true;
        }
      }

      index_ += 1;
    }

    if (token_started) {
      return str_.substr(start, end - start);
    }

    return "";
  }

 private:
  std::string str_;
  size_t index_;
};

#define PARSE_ERROR(reason, spl)              \
  do {                                        \
    LOG(ERROR) << "Error at parsing " << spl; \
    LOG(ERROR) << "Reason: " << reason;       \
    return false;                             \
  } while (0)

bool SPLParser::Parse(const std::string& spl,
                      std::vector<SPLStatement>& statements) {
  DLOG(INFO) << "Parsing:" << std::endl << spl;

  Tokenizer tokenizer(spl);

  while (tokenizer.HasNext()) {
    // Parse one statement at a time
    std::string t1 = tokenizer.NextToken();

    if (t1 == "\n" || t1 == "") continue;

    std::string t2 = tokenizer.NextToken();
    if (t2 == "") PARSE_ERROR("Expect token after " + t1 + "||", spl);

    SPLStatement statement;

    if (t2 == "=") {
      statement.processor_name = t1;
      // Processor creation
      std::string t3 = tokenizer.NextToken();
      if (t3 == "") PARSE_ERROR("Expect 'processor' or 'camera'", spl);
      if (t3 == "camera" || t3 == "processor") {
        statement.statement_type = SPL_STATEMENT_PROCESSOR;

        std::string t4 = tokenizer.NextToken();
        if (t4 != "(") PARSE_ERROR("Expect ( after " + t3, spl);
        std::string t5 = tokenizer.NextToken();
        if (t5 == "") PARSE_ERROR("Expect token after " + t4, spl);

        // Special condition for camera processor, where the token here is the
        // camera name instead of processor type.
        if (t3 == "camera") {
          statement.processor_type = PROCESSOR_TYPE_CAMERA;
          statement.params.insert({"camera_name", t5});
        } else {
          statement.processor_type = GetProcessorTypeByString(t5);
        }

        std::string t6 = tokenizer.NextToken();
        if (t6 == ")") {
          // Done
          statements.push_back(statement);
        } else if (t6 == ",") {
          while (t6 == ",") {
            std::string key = tokenizer.NextToken();
            if (key == "")
              PARSE_ERROR("Expect param key name after " + t6, spl);
            std::string equal = tokenizer.NextToken();
            if (equal != "=") PARSE_ERROR("Expect = after " + key, spl);
            std::string value = tokenizer.NextToken();
            if (value == "") PARSE_ERROR("Expect value after =", spl);
            statement.params.insert({key, value});
            t6 = tokenizer.NextToken();
          }
          if (t6 == ")") {
            statements.push_back(statement);
          } else {
            PARSE_ERROR("Bracket ( is not closed", spl);
          }
        } else {
          PARSE_ERROR("Unexpected token: " + t6, spl);
        }
      } else {
        PARSE_ERROR("Unexpected token: " + t3, spl);
      }
    } else if (t2 == "[") {
      // Connect
      statement.statement_type = SPL_STATEMENT_CONNECT;
      statement.lhs_processor_name = t1;

      std::string t3 = tokenizer.NextToken();
      if (t3 == "") PARSE_ERROR("Expect stream name for " + t1, spl);
      statement.lhs_stream_name = t3;

      std::string t4 = tokenizer.NextToken();
      if (t4 != "]") PARSE_ERROR("Bracket [ is not closed", spl);

      std::string equal = tokenizer.NextToken();
      if (equal != "=") PARSE_ERROR("RHS stream is not specified", spl);

      std::string t5 = tokenizer.NextToken();
      if (t5 == "") PARSE_ERROR("RHS processor name is not valid", spl);
      statement.rhs_processor_name = t5;

      std::string t6 = tokenizer.NextToken();
      if (t6 != "[") PARSE_ERROR("RHS stream is not specified", spl);

      std::string t7 = tokenizer.NextToken();
      if (t7 == "") PARSE_ERROR("RHS stream name is not valid", spl);
      statement.rhs_stream_name = t7;

      std::string t8 = tokenizer.NextToken();
      if (t8 != "]") PARSE_ERROR("Bracket [ is not closed", spl);

      statements.push_back(statement);
    } else {
      PARSE_ERROR("Expect = or [ after " + t1, spl);
    }

    std::string t_last = tokenizer.NextToken();
    if (t_last != "\n" && t_last != "") {
      PARSE_ERROR("Expect a newline after a statement", spl);
    }
  }  // while

  return true;
}
