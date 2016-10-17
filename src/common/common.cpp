//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include "common.h"

toml::Value ParseTomlFromFile(const string &filepath) {
  std::ifstream ifs(filepath);
  CHECK(!ifs.fail()) << "Can't open file " << filepath << " for read";
  toml::ParseResult pr = toml::parse(ifs);
  CHECK(pr.valid()) << "Toml file " << filepath
                    << " is not a valid toml file:" << std::endl
                    << pr.errorReason;
  return pr.value;
}
