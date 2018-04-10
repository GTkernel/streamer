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

#ifndef STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_
#define STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_

#include <mutex>

#include <tensorflow/core/public/session.h>

#include <Eigen/Dense>

#include "common/types.h"
#include "processor/processor.h"

#include "processor/fv_gen.h"

typedef struct query_t {
  int query_id;
  // indices holds a mask of which images this query uses in the queries_
  // matrix. This could be made more efficient by using an Eigen::VectorXd to
  // hold a binary coefficient matrix used to take a linear combination of rows
  // in queries_.
  float threshold;
  std::vector<int> matches;
  std::unique_ptr<tensorflow::Session> classifier;
  std::string unique_identifier;
  FvSpec fv_spec;
} query_t;

class ImageMatch : public Processor {
 public:
  ImageMatch(unsigned int batch_size = 1);

  static std::shared_ptr<ImageMatch> Create(const FactoryParamsType& params);

  // Add real query with Micro Classifier
  void AddQuery(const std::string& model_path, std::string layer_name,
                float threshold = 0.5, int xmin = 0, int ymin = 0, int xmax = 0,
                int ymax = 0, bool flat = true);
  void SetSink(StreamPtr stream);
  using Processor::SetSink;
  void SetSource(StreamPtr stream);
  using Processor::SetSource;
  StreamPtr GetSink();
  using Processor::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  // SetClassifier is a helper function to add/replace a Micro Classifier for
  // an existing query
  void SetClassifier(query_t* current_query, const std::string& model_path);
  // Batch size should be above 0
  unsigned int batch_size_;
  // cur_batch_frames holds the actual frames in the batch
  std::vector<std::unique_ptr<Frame>> frames_batch_;
  // query_data_ stores information related to the query
  // as well as state data regarding the most recent run of the
  // Micro Classifier related to this query
  std::unordered_map<int, query_t> query_data_;
  // Lock to prevent race conditions when adding/modifying queries
  std::mutex query_guard_;
};

#endif  // STREAMER_PROCESSOR_IMAGEMATCH_IMAGEMATCH_H_
