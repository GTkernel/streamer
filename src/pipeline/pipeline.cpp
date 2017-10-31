//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#include <unordered_map>

#include "json/src/json.hpp"

#include "common/types.h"
#include "pipeline/pipeline.h"
#include "pipeline/spl_parser.h"
#include "processor/processor_factory.h"

constexpr auto DEFAULT_SINK_NAME = "output";

std::shared_ptr<Pipeline> Pipeline::ConstructPipeline(
    const std::vector<SPLStatement>& spl_statements) {
  std::shared_ptr<Pipeline> pipeline(new Pipeline);

  for (const auto& stmt : spl_statements) {
    switch (stmt.statement_type) {
      case SPL_STATEMENT_PROCESSOR: {
        std::shared_ptr<Processor> processor;
        processor = ProcessorFactory::Create(stmt.processor_type, stmt.params);
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

std::shared_ptr<Pipeline> Pipeline::ConstructPipeline(nlohmann::json json) {
  nlohmann::json processors = json["processors"];
  std::cout << "Pipeline:\n" + json.dump(4) << std::endl;

  std::shared_ptr<Pipeline> pipeline(new Pipeline);
  std::unordered_map<std::string, std::string> id_to_type;

  // First pass to create all processors
  for (auto& processor_spec : processors) {
    std::string processor_name = processor_spec["processor_name"];
    std::string processor_type_str = processor_spec["processor_type"];
    std::unordered_map<std::string, nlohmann::json> processor_parameters_json =
        processor_spec["parameters"]
            .get<std::unordered_map<std::string, nlohmann::json>>();
    std::unordered_map<std::string, std::string> processor_parameters;
    for (auto const& pair : processor_parameters_json) {
      std::string key = pair.first;
      std::string value = pair.second.get<std::string>();
      processor_parameters[key] = value;
    }
    ProcessorType processor_type = GetProcessorTypeByString(processor_type_str);
    FactoryParamsType params = (FactoryParamsType)processor_parameters;

    std::cout << "Creating processor \"" << processor_name << "\" of type \""
              << processor_type_str << "\"" << std::endl;
    std::shared_ptr<Processor> processor =
        ProcessorFactory::Create(processor_type, params);
    pipeline->processors_.insert({processor_name, processor});
    pipeline->dependency_graph_.insert({processor.get(), {}});
    pipeline->reverse_dependency_graph_.insert({processor.get(), {}});
    id_to_type[processor_name] = processor_type_str;
  }

  // Second pass to create all processors
  for (auto& processor_spec : processors) {
    auto inputs_it = processor_spec.find("inputs");
    if (inputs_it != processor_spec.end()) {
      std::unordered_map<std::string, nlohmann::json> inputs =
          processor_spec["inputs"]
              .get<std::unordered_map<std::string, nlohmann::json>>();

      std::string cur_proc_id = processor_spec["processor_name"];
      std::shared_ptr<Processor> cur_processor =
          pipeline->GetProcessor(cur_proc_id);

      for (const auto input : inputs) {
        std::string src = input.first;
        std::string stream_id = input.second;
        size_t i = stream_id.find(":");
        std::string src_proc_id;
        std::string sink;
        if (i == std::string::npos) {
          // Use default sink name
          src_proc_id = stream_id;
          sink = DEFAULT_SINK_NAME;
        } else {
          // Use custom sink name
          src_proc_id = stream_id.substr(0, i);
          sink = stream_id.substr(i + 1, stream_id.length());
        }
        std::shared_ptr<Processor> src_processor =
            pipeline->GetProcessor(src_proc_id);

        cur_processor->SetSource(src, src_processor->GetSink(sink));
        pipeline->dependency_graph_[cur_processor.get()].insert(
            src_processor.get());
        pipeline->reverse_dependency_graph_[src_processor.get()].insert(
            cur_processor.get());

        std::cout << "Connected source \"" << src << "\" of processor \""
                  << cur_proc_id << "\" to the sink \"" << sink
                  << "\" from processor \"" << src_proc_id << "\"" << std::endl;
      }
    }
  }

  return pipeline;
}

Pipeline::Pipeline() {}

std::unordered_map<std::string, std::shared_ptr<Processor>>
Pipeline::GetProcessors() {
  return processors_;
}

std::shared_ptr<Processor> Pipeline::GetProcessor(const std::string& name) {
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
