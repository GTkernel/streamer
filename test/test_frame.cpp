
#include <gtest/gtest.h>

#include "stream/frame.h"

std::vector<std::string> tags_master = {"tag1", "tag2"};
std::vector<Rect> bboxes_master = {Rect(1, 2, 3, 4), Rect(5, 6, 7, 8)};

// Verifies that a MetadataFrame containing tags is correctly converted to a
// JSON object by MetadataFrame::ToJson. The resulting JSON should look like:
//   {
//     "MetadataFrame": {
//       "bboxes": [
//       ]
//       "tags": [
//         "tag1",
//         "tag2"
//       ]
//     }
//   }
TEST(TestFrame, TestMetadataFrameToJsonTags) {
  MetadataFrame md(tags_master);
  nlohmann::json j = md.ToJson();
  nlohmann::json md_j = j.at("MetadataFrame");

  std::vector<std::string> tags =
      md_j.at("tags").get<std::vector<std::string>>();
  int num_tags = tags.size();
  EXPECT_EQ(num_tags, tags_master.size());
  for (int i = 0; i < num_tags; i++) {
    EXPECT_EQ(tags[i], tags_master[i]);
  }

  std::vector<nlohmann::json> bboxes =
      md_j.at("bboxes").get<std::vector<nlohmann::json>>();
  EXPECT_EQ(bboxes.size(), 0);
}

// Verifies that MetadataFrame::MetadataFrame(nlohmann::json) creates a
// properly-formatted MetadataFrame object from a JSON object containing only
// tags. See TestMetadataFrameToJsonTags for details on the format of the JSON
// object.
TEST(TestFrame, TestJsonToMetadataFrameTags) {
  nlohmann::json md_j;
  md_j["tags"] = tags_master;
  std::vector<nlohmann::json> bboxes_j;
  md_j["bboxes"] = bboxes_j;
  nlohmann::json j;
  j["MetadataFrame"] = md_j;
  MetadataFrame md(j);

  std::vector<std::string> tags = md.GetTags();
  int num_tags = tags.size();
  EXPECT_EQ(num_tags, tags_master.size());
  for (int i = 0; i < num_tags; i++) {
    EXPECT_EQ(tags[i], tags_master[i]);
  }

  std::vector<Rect> bboxes = md.GetBboxes();
  EXPECT_EQ(bboxes.size(), 0);
}

// Verifies that a MetadataFrame containing bounding boxes is correctly
// converted to a JSON object by MetadataFrame::ToJson. The resulting JSON
// should look like:
//   {
//     "MetadataFrame": {
//       "bboxes": [
//         {
//           "Rect": {
//             "px": 1,
//             "py": 2,
//             "width": 3,
//             "height": 4
//           }
//         },
//         {
//           "Rect": {
//             "px": 5,
//             "py": 6,
//             "width": 7,
//             "height": 8
//           }
//         }
//       ]
//       "tags": [
//       ]
//     }
//   }
TEST(TestFrame, TestMetadataFrameToJsonBboxes) {
  MetadataFrame md(bboxes_master);
  nlohmann::json j = md.ToJson();
  nlohmann::json md_j = j.at("MetadataFrame");

  std::vector<std::string> tags =
      md_j.at("tags").get<std::vector<std::string>>();
  EXPECT_EQ(tags.size(), 0);

  std::vector<nlohmann::json> bboxes =
      md_j.at("bboxes").get<std::vector<nlohmann::json>>();
  int num_bboxes = bboxes.size();
  EXPECT_EQ(num_bboxes, bboxes_master.size());
  for (int i = 0; i < num_bboxes; i++) {
    Rect r(bboxes[i]);
    EXPECT_EQ(r, bboxes_master[i]);
  }
}

// Verifies that MetadataFrame::MetadataFrame(nlohmann::json) creates a
// properly-formatted MetadataFrame object from a JSON object containing only
// bounding boxes. See TestMetadataFrameToJsonTags for details on the format of
// the JSON object.
TEST(TestFrame, TestJsonToMetadataFrameBboxes) {
  std::vector<nlohmann::json> bboxes_j;
  for (auto bbox : bboxes_master) {
    bboxes_j.push_back(bbox.ToJson());
  }

  nlohmann::json md_j;
  std::vector<std::string> tags_j;
  md_j["tags"] = tags_j;
  md_j["bboxes"] = bboxes_j;
  nlohmann::json j;
  j["MetadataFrame"] = md_j;
  MetadataFrame md(j);

  std::vector<std::string> tags = md.GetTags();
  EXPECT_EQ(tags.size(), 0);

  std::vector<Rect> bboxes = md.GetBboxes();
  int num_bboxes = bboxes.size();
  EXPECT_EQ(num_bboxes, bboxes_master.size());
  for (int i = 0; i < num_bboxes; i++) {
    EXPECT_EQ(bboxes[i], bboxes_master[i]);
  }
}