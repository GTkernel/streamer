#ifndef STREAMER_PROCESSOR_DB_FILE_WRITER_H_
#define STREAMER_PROCESSOR_DB_FILE_WRITER_H_

#include <chrono>
#include <stdlib.h>
#include <fstream>

#include "common/types.h"
#include "processor/processor.h"
#include "litesql.hpp"

class DBFileWriter : public Processor {
 public:
  /**
   * @brief DBFileWriter constructor
   * @param root_dir The name of the file.
   */
  DBFileWriter(const string& root_dir);

  static std::shared_ptr<DBFileWriter> Create(const FactoryParamsType& params);
  void SetRate(int rate);

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  string root_dir_;
  // Match types to avoid compiler warnings
  unsigned long long expected_timestamp_;
  unsigned long long delay_ms_;
};

#endif  // STREAMER_PROCESSOR_DB_FILE_WRITER_H_
