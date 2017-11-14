//
// Created by Ran Xian (xranthoar@gmail.com) on 10/26/16.
//

#include <gtest/gtest.h>
#include "utils/string_utils.h"

TEST(STRING_UTILS_TEST, GET_IP_ADDR_FROM_STRING_TEST) {
  std::string ip_addr = "1.2.3.4";
  unsigned ip_val = GetIPAddrFromString(ip_addr);
  EXPECT_EQ(ip_val, 0x01020304);
}
