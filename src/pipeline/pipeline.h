//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#ifndef STREAMER_PIPELINE_PIPELINE_H_
#define STREAMER_PIPELINE_PIPELINE_H_

#include "common/common.h"
#include "processor/processor.h"
#include "spl_parser.h"

class Pipeline {
 public:
  static std::shared_ptr<Pipeline> ConstructPipeline(
      const std::vector<SPLStatement>& spl_statements);
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
  std::unordered_map<string, std::shared_ptr<Processor>> processors_;
  // I depend on who
  std::unordered_map<Processor*, std::unordered_set<Processor*>>
      dependency_graph_;
  // Who depends on me
  std::unordered_map<Processor*, std::unordered_set<Processor*>>
      reverse_dependency_graph_;
};

#endif  // STREAMER_PIPELINE_PIPELINE_H_
