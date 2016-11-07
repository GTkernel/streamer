//
// Created by Ran Xian (xranthoar@gmail.com) on 11/5/16.
//

#include "pipeline.h"
#include "processor/processor_factory.h"

std::shared_ptr<Pipeline> Pipeline::ConstructPipeline(
    const std::vector<SPLStatement> &spl_statements) {
  std::shared_ptr<Pipeline> pipeline(new Pipeline);

  for (auto &stmt : spl_statements) {
    switch (stmt.statement_type) {
      case SPL_STATEMENT_PROCESSOR: {
        std::shared_ptr<Processor> processor;
        processor =
            ProcessorFactory::CreateInstance(stmt.processor_type, stmt.params);
        pipeline->processors_.insert({stmt.processor_name, processor});
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

std::shared_ptr<Processor> Pipeline::GetProcessor(const string &name) {
  CHECK(processors_.count(name) != 0) << "Has no processor named: " << name;

  return processors_[name];
}
