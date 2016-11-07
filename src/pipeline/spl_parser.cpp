//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#include "spl_parser.h"

SPLParser::SPLParser() {}

// TODO: this is definitely going to be rewrite..
class Tokenizer {
 public:
  Tokenizer(const string &str) : str_(str), index_(0) {}

  bool HasNext() { return index_ != str_.length(); }

  string NextToken() {
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
          return string(1, ch);
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
  string str_;
  size_t index_;
};

#define PARSE_ERROR(reason, spl)              \
  do {                                        \
    LOG(ERROR) << "Error at parsing " << spl; \
    LOG(ERROR) << "Reason: " << reason;       \
    return false;                             \
  } while (0)

bool SPLParser::Parse(const string &spl,
                      std::vector<SPLStatement> &statements) {
  DLOG(INFO) << "Parsing:" << std::endl << spl;

  Tokenizer tokenizer(spl);

  while (tokenizer.HasNext()) {
    // Parse one statement at a time
    string t1 = tokenizer.NextToken();

    if (t1 == "\n") continue;

    string t2 = tokenizer.NextToken();
    if (t2 == "") PARSE_ERROR("Expect token after " + t2, spl);

    SPLStatement statement;

    if (t2 == "=") {
      statement.processor_name = t1;
      // Processor creation
      string t3 = tokenizer.NextToken();
      if (t3 == "") PARSE_ERROR("Expect 'processor' or 'camera'", spl);
      if (t3 == "camera" || t3 == "processor") {
        statement.statement_type = SPL_STATEMENT_PROCESSOR;

        string t4 = tokenizer.NextToken();
        if (t4 != "(") PARSE_ERROR("Expect ( after " + t3, spl);
        string t5 = tokenizer.NextToken();
        if (t5 == "") PARSE_ERROR("Expect token after " + t4, spl);

        // Special condition for camera processor, where the token here is the
        // camera name instead of processor type.
        if (t3 == "camera") {
          statement.processor_type = PROCESSOR_TYPE_CAMERA;
          statement.params.insert({"camera_name", t5});
        } else {
          statement.processor_type = GetProcessorTypeByString(t5);
        }

        string t6 = tokenizer.NextToken();
        if (t6 == ")") {
          // Done
          statements.push_back(statement);
        } else if (t6 == ",") {
          while (t6 == ",") {
            string key = tokenizer.NextToken();
            if (key == "")
              PARSE_ERROR("Expect param key name after " + t6, spl);
            string equal = tokenizer.NextToken();
            if (equal != "=") PARSE_ERROR("Expect = after " + key, spl);
            string value = tokenizer.NextToken();
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

      string t3 = tokenizer.NextToken();
      if (t3 == "") PARSE_ERROR("Expect stream name for " + t1, spl);
      statement.lhs_stream_name = t3;

      string t4 = tokenizer.NextToken();
      if (t4 != "]") PARSE_ERROR("Bracket [ is not closed", spl);

      string equal = tokenizer.NextToken();
      if (equal != "=") PARSE_ERROR("RHS stream is not specified", spl);

      string t5 = tokenizer.NextToken();
      if (t5 == "") PARSE_ERROR("RHS processor name is not valid", spl);
      statement.rhs_processor_name = t5;

      string t6 = tokenizer.NextToken();
      if (t6 != "[") PARSE_ERROR("RHS stream is not specified", spl);

      string t7 = tokenizer.NextToken();
      if (t7 == "") PARSE_ERROR("RHS stream name is not valid", spl);
      statement.rhs_stream_name = t7;

      string t8 = tokenizer.NextToken();
      if (t8 != "]") PARSE_ERROR("Bracket [ is not closed", spl);

      statements.push_back(statement);
    } else {
      PARSE_ERROR("Expect = or [ after " + t1, spl);
    }

    string t_last = tokenizer.NextToken();
    if (t_last != "\n" && t_last != "") {
      PARSE_ERROR("Expect a newline after a statement", spl);
    }
  }  // while

  return true;
}
