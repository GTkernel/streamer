#ifndef STREAMER_CAFFE_MTCNN_H
#define STREAMER_CAFFE_MTCNN_H

// caffe
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

// c++
#include <string>
#include <vector>
#include <fstream>
// opencv
#include <opencv2/opencv.hpp>
// boost
#include "boost/make_shared.hpp"

#include "common/common.h"
#include "model/model.h"
#include "processor.h"

using caffe::Datum;
using caffe::Blob;
using caffe::Net;
using caffe::Caffe;
using caffe::TEST;
using caffe::MemoryDataLayer;

class MTCNN {
 public:
  MTCNN(ModelDescription& model_description);
  void Detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo, int minSize, double* threshold, double factor);

 private:
  bool CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum);
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels,Blob<float>* input_layer,
          const int height,const int width);
  void SetMean();
  void GenerateBoundingBox( Blob<float>* confidence,Blob<float>* reg,
          float scale,float thresh,int image_width,int image_height);
  void ClassifyFace(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_single,
        boost::shared_ptr<Net<float> >& net,double thresh,char netName);
  void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_single,
        boost::shared_ptr<Net<float> >& net, double thresh, char netName);
  std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes,float thresh,char methodType);
  void Bbox2Square(std::vector<FaceInfo>& bboxes);
  void Padding(int img_w, int img_h);
  std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);
  void RegressPoint(const std::vector<FaceInfo>& faceInfo);

 private:
  boost::shared_ptr<Net<float> > PNet_;
  boost::shared_ptr<Net<float> > RNet_;
  boost::shared_ptr<Net<float> > ONet_;

  // x1,y1,x2,t2 and score
  std::vector<FaceInfo> condidate_rects_;
  std::vector<FaceInfo> total_boxes_;
  std::vector<FaceInfo> regressed_rects_;
  std::vector<FaceInfo> regressed_pading_;

  std::vector<cv::Mat> crop_img_;
  int curr_feature_map_w_;
  int curr_feature_map_h_;
  int num_channels_;
};

class MtcnnFaceDetector : public Processor {
public:
  MtcnnFaceDetector(const ModelDescription& model_description, int min_size);
  virtual ProcessorType GetType() override;

protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

private:
  ModelDescription model_description_;
  std::unique_ptr<MTCNN> detector_;
  double threshold_[3];
  double factor_;
  int minSize_;
};

#endif  // STREAMER_CAFFE_MTCNN_H
