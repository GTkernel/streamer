//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#include "pipeline.h"
#include "processor/processor_factory.h"

std::shared_ptr<Pipeline> Pipeline::ConstructPipeline(
    const std::vector<SPLStatement>& spl_statements) {
  std::shared_ptr<Pipeline> pipeline(new Pipeline);

  for (const auto& stmt : spl_statements) {
    switch (stmt.statement_type) {
      case SPL_STATEMENT_PROCESSOR: {
        std::shared_ptr<Processor> processor;
        processor =
            ProcessorFactory::CreateInstance(stmt.processor_type, stmt.params);
        pipeline->processors_.insert({stmt.processor_name, processor});
        pipeline->dependency_graph_.insert({processor.get(), {}});
        pipeline->reverse_dependency_graph_.insert({processor.get(), {}});
        break;
      }
      case SPL_STATEMENT_CONNECT: {
        // Check parameters
        CHECK(pipeline->processors_.count(stmt.lhs_processor_name) != 0)
            << "Processor with name: " << stmt.lhs_processor_name
            << " is not declared";
        CHECK(pipeline->processors_.count(stmt.rhs_processor_name) != 0)
            << "Processor with name: " << stmt.rhs_processor_name
            << " is not declared";
        auto lhs_processor = pipeline->GetProcessor(stmt.lhs_processor_name);
        auto rhs_processor = pipeline->GetProcessor(stmt.rhs_processor_name);

        lhs_processor->SetSource(stmt.lhs_stream_name,
                                 rhs_processor->GetSink(stmt.rhs_stream_name));

        pipeline->dependency_graph_[lhs_processor.get()].insert(
            rhs_processor.get());
        pipeline->reverse_dependency_graph_[rhs_processor.get()].insert(
            lhs_processor.get());
        break;
      }
      default:
        LOG(FATAL) << "SPL statement of unknown type encountered";
    }
  }

  return pipeline;
}

Pipeline::Pipeline() {}

std::unordered_map<string, std::shared_ptr<Processor>>
Pipeline::GetProcessors() {
  return processors_;
}

std::shared_ptr<Processor> Pipeline::GetProcessor(const string& name) {
  CHECK(processors_.count(name) != 0) << "Has no processor named: " << name;

  return processors_[name];
}

bool Pipeline::Start() {
  while (true) {
    bool some_processor_started = false;
    unsigned int started_processor = 0;
    for (const auto& itr : dependency_graph_) {
      auto processor = itr.first;
      // For all not yet start processors
      if (!processor->IsStarted()) {
        unsigned int started_dep = 0;
        for (const auto& dep : itr.second) {
          if (dep->IsStarted()) started_dep += 1;
        }
        if (itr.second.size() == started_dep) {
          // All dependencies have started
          processor->Start();
          some_processor_started = true;
        }
      } else {
        started_processor += 1;
      }
    }  // For
    if (started_processor == dependency_graph_.size()) {
      // All processors have started
      break;
    } else {
      if (some_processor_started) {
        // Scan again
      } else {
        // We have a loop
        LOG(ERROR) << "Has cycle in pipeline dependency graph!";
        return false;
      }
    }
  }  // While

  return true;
}

void Pipeline::Stop() {
  while (true) {
    unsigned int stopped_processor = 0;
    for (const auto& itr : reverse_dependency_graph_) {
      auto processor = itr.first;
      // For all not yet stop processors
      if (processor->IsStarted()) {
        unsigned int stopped_dep = 0;
        for (const auto& dep : itr.second) {
          if (!dep->IsStarted()) stopped_dep += 1;
        }
        if (itr.second.size() == stopped_dep) {
          // All dependencies have stopped
          processor->Stop();
        }
      } else {
        stopped_processor += 1;
      }
    }  // For
    if (stopped_processor == reverse_dependency_graph_.size()) {
      // All processors have stopped
      break;
    }
    // No need to check for cycle, it has been checked in Start()
  }  // While
}
