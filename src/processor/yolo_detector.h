/**
* Multi-target detection using YOLO
*
* @author Wendy Chin <wendy.chin@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#ifndef STREAMER_YOLO_DETECTOR_H
#define STREAMER_YOLO_DETECTOR_H
#include <set>
#include <caffe/caffe.hpp>
#include "common/common.h"
#include "model/model.h"
#include "processor.h"

namespace yolo {
	class Detector {
	public:
		Detector(const string& model_file,
				   const string& weights_file);
		cv::Mat Detect(cv::Mat& img);
		cv::Mat GetBox(std::vector<float>DetectionResult,float* pro_obj,
				int* idx_class, std::vector<std::vector<int>>& bboxs, 
				float thresh, cv::Mat img);

	private:
		std::shared_ptr<caffe::Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
	};
}



class YoloDetector : public Processor {
public:
	YoloDetector(const ModelDesc& model_desc,
		float idle_duration = 0.f);
	virtual ProcessorType GetType() override;

	
protected:
	virtual bool Init() override;
	virtual bool OnStop() override;
	virtual void Process() override;


private:
	std::string GetLabelName(int label) const;
	
private:
	ModelDesc model_desc_;
	std::unique_ptr<yolo::Detector> detector_;
	float idle_duration_;
	std::chrono::time_point<std::chrono::system_clock> last_detect_time_;

};

template<typename Dtype>
Dtype lap(Dtype x1_min, Dtype x1_max, Dtype x2_min, Dtype x2_max) {
	if (x1_min < x2_min) {
		if (x1_max < x2_min) {
			return 0;
		}
		else {
			if (x1_max > x2_min) {
				if (x1_max < x2_max) {
					return x1_max - x2_min;
				}
				else {
					return x2_max - x2_min;
				}
			}
			else {
				return 0;
			}
		}
	}
	else {
		if (x1_min < x2_max) {
			if (x1_max < x2_max)
				return x1_max - x1_min;
			else
				return x2_max - x1_min;
		}
		else {
			return 0;
		}
	}
}
	template int lap(int x1_min, int x1_max, int x2_min, int x2_max);
	template float lap(float x1_min, float x1_max, float x2_min, float x2_max);	

#endif // STREAMER_YOLO_DETECTOR_H
