
#include <gtest/gtest.h>

#include "json/json.hpp"

#include "common/types.h"

// Verifies that Rect::ToJson() produces a correctly-formatted JSON object The
// resulting JSON should look like this:
//   {
//     "Rect": {
//       "px": 1,
//       "py": 2,
//       "width": 3,
//       "height": 4
//     }
//   }
TEST(TestTypes, TestRectToJson) {
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;

  Rect r(a, b, c, d);
  nlohmann::json j = r.ToJson();
  nlohmann::json rect_j = j.at("Rect");

  EXPECT_EQ(rect_j.at("px").get<int>(), a);
  EXPECT_EQ(rect_j.at("py").get<int>(), b);
  EXPECT_EQ(rect_j.at("width").get<int>(), c);
  EXPECT_EQ(rect_j.at("height").get<int>(), d);
}

// Verifies that Rect::Rect(nlohmann::json) creates a properly-initialized Rect
// struct from a JSON object. See TestRectToJson for details on the format of
// the JSON object.
TEST(TestTypes, TestJsonToRect) {
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;

  nlohmann::json rect_j;
  rect_j["px"] = a;
  rect_j["py"] = b;
  rect_j["width"] = c;
  rect_j["height"] = d;
  nlohmann::json j;
  j["Rect"] = rect_j;
  Rect r(j);

  EXPECT_EQ(r.px, a);
  EXPECT_EQ(r.py, b);
  EXPECT_EQ(r.width, c);
  EXPECT_EQ(r.height, d);
}
