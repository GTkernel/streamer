/**
* Send metadata to the database
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#ifndef STREAMER_DB_WRITER_H
#define STREAMER_DB_WRITER_H

#ifdef USE_ATHENA
#include "client/AthenaClient.h"
#endif
#include "common/common.h"
#include "camera/camera.h"
#include "processor.h"

class DbWriter : public Processor {
  public:
    DbWriter(std::shared_ptr<Camera> camera, bool write_to_file, const std::string& athena_address);

  protected:
    virtual bool Init() override;
    virtual bool OnStop() override;
    virtual void Process() override;

  private:
    void WriteFile(const std::string& camera_id,
                   const std::vector<std::string>& uuids,
                   unsigned long timestamp,
                   const std::vector<string>& tags,
                   const std::vector<std::vector<double>>& struck_features);
    void WriteAthena(const std::string& camera_id,
                     const std::vector<std::string>& uuids,
                     unsigned long timestamp,
                     const std::vector<string>& tags,
                     const std::vector<std::vector<double>>& struck_features);

  private:
    std::shared_ptr<Camera> camera_;
    bool write_to_file_;
    std::ofstream ofs_;
    std::string athena_address_;
#ifdef USE_ATHENA
    std::unique_ptr<athena::AthenaClient> aclient_;
#endif
};

#endif // STREAMER_DB_WRITER_H
