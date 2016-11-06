//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#ifndef STREAMER_PARSER_H
#define STREAMER_PARSER_H

#include "common/common.h"
#include "pipeline.h"

/**
 * @brief Streamer pipeline parser
 */
class SPLParser {
 public:
  SPLParser();
  Pipeline Parse(const string &pipeline_description);
};

#endif  // STREAMER_PARSER_H
