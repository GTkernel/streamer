//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#include <boost/graph/topological_sort.hpp>
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
        pipeline->processor_names_.push_back(stmt.processor_name);
        boost::add_vertex(stmt.processor_name, pipeline->dependency_graph_);
        boost::add_vertex(stmt.processor_name,
                          pipeline->reverse_dependency_graph_);
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

        boost::add_edge_by_label(stmt.lhs_processor_name,
                                 stmt.rhs_processor_name,
                                 pipeline->dependency_graph_);
        boost::add_edge_by_label(stmt.rhs_processor_name,
                                 stmt.lhs_processor_name,
                                 pipeline->reverse_dependency_graph_);
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

    pipeline->processor_names_.push_back(processor_name);
    boost::add_vertex(processor_name, pipeline->dependency_graph_);
    boost::add_vertex(processor_name, pipeline->reverse_dependency_graph_);
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
        boost::add_edge_by_label(src_proc_id, cur_proc_id,
                                 pipeline->reverse_dependency_graph_);
        boost::add_edge_by_label(cur_proc_id, src_proc_id,
                                 pipeline->dependency_graph_);

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
  std::deque<Vertex> c;
  boost::topological_sort(dependency_graph_, std::front_inserter(c));
  std::cout << "Pipeline start order: ";
  for (auto& i : c) {
    auto name = processor_names_[i];
    std::cout << name << " ";
    processors_[name]->Start();
  }
  std::cout << std::endl;

  return true;
}

void Pipeline::Stop() {
  std::deque<Vertex> c;
  boost::topological_sort(reverse_dependency_graph_, std::front_inserter(c));
  std::cout << "Pipeline stop order: ";
  for (auto& i : c) {
    auto name = processor_names_[i];
    std::cout << name << " ";
    processors_[name]->Stop();
  }
  std::cout << std::endl;
}
