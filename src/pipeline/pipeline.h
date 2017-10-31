//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#ifndef STREAMER_PIPELINE_PIPELINE_H_
#define STREAMER_PIPELINE_PIPELINE_H_

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/labeled_graph.hpp>
#include <unordered_map>

#include "json/src/json.hpp"

#include "common/common.h"
#include "pipeline/spl_parser.h"
#include "processor/processor.h"

class Pipeline {
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>
      AdjList;
  typedef boost::labeled_graph<AdjList, std::string, boost::hash_mapS> Graph;
  typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

 public:
  static std::shared_ptr<Pipeline> ConstructPipeline(
      const std::vector<SPLStatement>& spl_statements);
  static std::shared_ptr<Pipeline> ConstructPipeline(nlohmann::json json);

  /**
   * @brief Initialize the pipeline from spl statements
   * @param spl_statements The spl statements used to construct the pipeline
   */
  Pipeline();
  /**
   * @brief Get a processor of the pipeline by its name
   * @return The processor
   */
  std::shared_ptr<Processor> GetProcessor(const string& name);

  std::unordered_map<string, std::shared_ptr<Processor>> GetProcessors();

  /**
   * @brief Start executing the pipeline
   */
  bool Start();

  /**
   * @brief Stop executing the pipeline
   */
  void Stop();

 private:
  std::unordered_map<std::string, std::shared_ptr<Processor>> processors_;
  std::vector<std::string> processor_names_;
  // I depend on who
  Graph dependency_graph_;
  // Who depends on me
  Graph reverse_dependency_graph_;
};

#endif  // STREAMER_PIPELINE_PIPELINE_H_
