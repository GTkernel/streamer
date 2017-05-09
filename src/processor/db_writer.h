/**
* Send metadata to the database
* 
* @author Tony Chen <xiaolongx.chen@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#ifndef STREAMER_DB_WRITER_H
#define STREAMER_DB_WRITER_H

#include "common/common.h"
#include "camera/camera.h"
#include "processor.h"

class DbWriter : public Processor {
  public:
    DbWriter(std::shared_ptr<Camera> camera);
    virtual ProcessorType GetType() override;

  protected:
    virtual bool Init() override;
    virtual bool OnStop() override;
    virtual void Process() override;

  private:
    std::shared_ptr<Camera> camera_;
};

#endif // STREAMER_DB_WRITER_H
