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

#ifndef STREAMER_PIPELINE_PIPELINE_H_
#define STREAMER_PIPELINE_PIPELINE_H_

#include <memory>
#include <string>
#include <unordered_map>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/labeled_graph.hpp>
#include <json/src/json.hpp>

#include "pipeline/spl_parser.h"
#include "processor/processor.h"

class Pipeline {
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>
      AdjList;
  typedef boost::labeled_graph<AdjList, std::string, boost::hash_mapS> Graph;
  typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

 public:
  Pipeline();

  // Creates a Pipeline from SPL statements.
  static std::shared_ptr<Pipeline> ConstructPipeline(
      const std::vector<SPLStatement>& spl_statements);
  // Creates a Pipeline from a JSON specification.
  static std::shared_ptr<Pipeline> ConstructPipeline(nlohmann::json json);

  // Returns the Processor with the specified name.
  std::shared_ptr<Processor> GetProcessor(const std::string& name);

  // Returns all of the Processors in this Pipeline.
  std::unordered_map<std::string, std::shared_ptr<Processor>> GetProcessors();

  // Starts executing the pipeline. Returns true if successful, or false if a
  // Processor failed to start.
  bool Start();

  // Stops executing the pipeline. Returns true if successful, or false if a
  // Processor failed to stop.
  bool Stop();

  // Get reverse dependency graph (the pipeline) in GraphViz format.
  const std::string GetGraph() const;

 private:
  std::unordered_map<std::string, std::shared_ptr<Processor>> processors_;
  std::vector<std::string> processor_names_;
  // Graph that tracks the Processors that each Processor depends on.
  Graph dependency_graph_;
  // Graph that tracks the Processors that depend on each Processor.
  Graph reverse_dependency_graph_;
};

#endif  // STREAMER_PIPELINE_PIPELINE_H_
