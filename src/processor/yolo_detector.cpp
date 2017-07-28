/**
* Multi-target detection using Caffe Yolo 
*
* @author Wendy Chin <wendy.chin@intel.com>
* @author Shao-Wen Yang <shao-wen.yang@intel.com>
*/

#include "common/context.h"
#include "yolo_detector.h"
#include "model/model_manager.h"


namespace yolo {
	Detector::Detector(const string& model_file,
					   const string& weights_file) {
		// Set Caffe backend
		int desired_device_number = Context::GetContext().GetInt(DEVICE_NUMBER);

		if (desired_device_number == DEVICE_NUMBER_CPU_ONLY) {
			LOG(INFO) << "Use device: " << desired_device_number << "(CPU)";
			caffe::Caffe::set_mode(caffe::Caffe::CPU);
		}
		else {
#ifdef USE_CUDA
			std::vector<int> gpus;
			GetCUDAGpus(gpus);

			if (desired_device_number < gpus.size()) {
				// Device exists
				LOG(INFO) << "Use GPU with device ID " << desired_device_number;
				caffe::Caffe::SetDevice(desired_device_number);
				caffe::Caffe::set_mode(caffe::Caffe::GPU);
			}
			else {
				LOG(FATAL) << "No GPU device: " << desired_device_number;
			}
#elif USE_OPENCL
			std::vector<int> gpus;
			int count = caffe::Caffe::EnumerateDevices();

			if (desired_device_number < count) {
				// Device exists
				LOG(INFO) << "Use GPU with device ID " << desired_device_number;
				caffe::Caffe::SetDevice(desired_device_number);
				caffe::Caffe::set_mode(caffe::Caffe::GPU);
			}
			else {
				LOG(FATAL) << "No GPU device: " << desired_device_number;
			}
#else
			LOG(FATAL) << "Compiled in CPU_ONLY mode but have a device number "
				"configured rather than -1";
#endif
		}

		/* Load the network. */
		net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
		net_->CopyTrainedLayersFromBinaryProto(weights_file);

		CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
		CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

		caffe::Blob<float>* input_layer = net_->input_blobs()[0];
		num_channels_ = input_layer->channels();
		CHECK(num_channels_ == 3 || num_channels_ == 1)
			<< "Input layer should have 1 or 3 channels.";
		input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	}
	

	cv::Mat Detector::Detect(const cv::Mat& img){
		caffe::Blob<float>* input_layer = net_->input_blobs()[0];
		int width, height;
		width = input_layer->width();
		height = input_layer->height();
		int size = width*height;
		cv::Mat image_resized;
		cv::resize(img, image_resized, cv::Size(height, width));
		
		float* input_data = input_layer->mutable_cpu_data();
		int temp, idx;
		for (int i = 0; i < height; ++i) {
			uchar* pdata = image_resized.ptr<uchar>(i);
			for (int j = 0; j < width; ++j) {
				temp = 3 * j;
				idx = i*width + j;
				input_data[idx] = (pdata[temp + 2] / 127.5) - 1;
				input_data[idx + size] = (pdata[temp + 1] / 127.5) - 1;
				input_data[idx + 2 * size] = (pdata[temp + 0] / 127.5) - 1;
			}
		}

		net_->Forward();

		caffe::Blob<float>* output_layer = net_->output_blobs()[0];
		const float* begin = output_layer->cpu_data();
		const float* end = begin + output_layer->channels();
		std::vector<float> DetectionResult(begin, end);
		
		
		std::vector<std::vector<int> > bboxs;
		float pro_obj[49][2];
		int idx_class[49];	
     
        cv::Mat FinalImage= GetBox(DetectionResult,&pro_obj[0][0],idx_class,bboxs,0.2,img);
		return FinalImage;
		}
		

	cv::Mat Detector::GetBox(std::vector<float>DetectionResult,float* pro_obj,
				int* idx_class, std::vector<std::vector<int>>& bboxs, 
				float thresh, cv::Mat img){

		char *labelname[] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

		float overlap;
		float overlap_thresh=0.4;
		float pro_class[49];
		int idx;
		int idx2;	
		float max_idx;
		float max;

                
		for (int i = 0; i < 7; ++i) {
			for (int j = 0; j < 7; ++j) {
				max = 0;
				max_idx = 0;
				idx2 = 20 * (i * 7 + j);
				for (int k = 0; k < 20; ++k) {
					if (DetectionResult[idx2 + k] > max) {
						max = DetectionResult[idx2 + k];
						max_idx = k + 1;
					}
				}
				idx_class[i * 7 + j] = max_idx;
				pro_class[i * 7 + j] = max;
                                
				pro_obj[(i * 7 + j) *2]=max*DetectionResult[7 * 7 * 20 + (i * 7 + j) * 2];
				pro_obj[(i * 7 + j)*2+1]=max*DetectionResult[7 * 7 * 20 + (i * 7 + j) * 2 + 1];
				                

			}
		}
		
		std::vector<int> bbox;
		int x_min, x_max, y_min, y_max;
		float x, y, w, h;
		for (int i = 0; i < 7; ++i) {
			for (int j = 0; j < 7; ++j) {
				for (int k = 0; k < 2; ++k) {
					if (pro_obj[(i * 7 + j) * 2 + k] > thresh) {
						//std::cout << "(" << i << "," << j << "," << k << ")" << " prob="<<pro_obj[(i*7+j)*2 + k] << " class="<<idx_class[i*7+j]<<std::endl;
						idx = 49 * 20 + 49 * 2 + ((i * 7 + j) * 2 + k) * 4;
						x = img.cols*(DetectionResult[idx++] + j) / 7;
						y = img.rows*(DetectionResult[idx++] + i) / 7;
						w = img.cols*DetectionResult[idx] * DetectionResult[idx++];
						h = img.rows*DetectionResult[idx] * DetectionResult[idx];
						x_min = x - w / 2;
						y_min = y - h / 2;
						x_max = x + w / 2;
						y_max = y + h / 2;
						bbox.clear();
						bbox.push_back(idx_class[i * 7 + j]);
						bbox.push_back(x_min);
						bbox.push_back(y_min);
						bbox.push_back(x_max);
						bbox.push_back(y_max);
						bbox.push_back(int(pro_obj[(i * 7 + j) * 2 + k] * 100));
						bboxs.push_back(bbox);
					}
				}
			}
		}
		
	
		std::vector<bool> mark(bboxs.size(), true);
				for (int i = 0; i < bboxs.size(); ++i) {
					for (int j = i + 1; j < bboxs.size(); ++j) {
						int overlap_x = lap(bboxs[i][0], bboxs[i][2], bboxs[j][0], bboxs[j][2]);
						int overlap_y = lap(bboxs[i][1], bboxs[i][3], bboxs[j][1], bboxs[j][3]);
						overlap = (overlap_x*overlap_y)*1.0 / ((bboxs[i][0] - bboxs[i][2])*(bboxs[i][1] - bboxs[i][3]) + (bboxs[j][0] - bboxs[j][2])*(bboxs[j][1] - bboxs[j][3]) - (overlap_x*overlap_y));
						if (overlap > overlap_thresh) {
							if (bboxs[i][4] > bboxs[j][4]) {
								mark[j] = false;
							}
							else {
								mark[i] = false;
							}
						}
					}
				}
		            						
				for (int i = 0; i < bboxs.size(); ++i) {
					if (mark[i]) {
						cv::Point point1(bboxs[i][1], bboxs[i][2]);
						cv::Point point2(bboxs[i][3], bboxs[i][4]);
						cv::rectangle(img, cv::Rect(point1, point2), cv::Scalar(0, bboxs[i][0] / 20.0 * 225, 255), bboxs[i][5] / 8);
						char ch[100];
						sprintf(ch, "%s %.2f", labelname[bboxs[i][0] - 1], bboxs[i][5] * 1.0 / 100);
						std::string temp(ch);
						cv::putText(img, temp, point1, CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));
					}
				}

				return img;
		}	
}
		
	
YoloDetector::YoloDetector(const ModelDesc& model_desc,
	float idle_duration)
	: Processor(PROCESSOR_TYPE_YOLO_DETECTOR, { "input" }, { "output" }),
	model_desc_(model_desc),
	idle_duration_(idle_duration){}

bool YoloDetector::Init() {
	std::string model_file = model_desc_.GetModelDescPath();
	std::string weights_file = model_desc_.GetModelParamsPath();
	LOG(INFO) << "model_file: " << model_file;
	LOG(INFO) << "weights_file: " << weights_file;

	detector_.reset(new yolo::Detector(model_file,weights_file));

	LOG(INFO) << "YoloDetector initialized";
	return true;
}

bool YoloDetector::OnStop() {
	return true;
}

void YoloDetector::Process() {
	Timer timer;
	timer.Start();

  auto frame = GetFrame("input");

	auto now = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = now - last_detect_time_;
	
	if (diff.count() >= idle_duration_) {
    const cv::Mat& image = frame->GetValue<cv::Mat>("image");
		cv::Mat DetectionOutput= detector_->Detect(image);
		LOG(INFO) << "Yolo detection took " << timer.ElapsedMSec() << " ms";
		cv::imshow("Image",DetectionOutput);
		cv::waitKey(1);
	} else {
    PushFrame("output", std::move(frame));
	}
}
