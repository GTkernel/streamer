//
// Created by Ran Xian on 8/4/16.
//

#include "mxnet_classifier.h"
#include "utils.h"
#include <fstream>

/**
 * @brief A buffer representation of a file. The buffer is automatically deallocated when the
 * buffer file object is outout of scope.
 * 
 * @param file_path The path to the file to be buffered.
 */
class BufferFile {
 public :
  /**
   * @brief Initialize from a file.
   */
  explicit BufferFile(std::string file_path)
      :file_path_(file_path) {
    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    CHECK(ifs) << "Can't open the file. Please check " << file_path;

    ifs.seekg(0, std::ios::end);
    length_ = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    LOG(INFO) << file_path.c_str() << " ... "<< length_ << " bytes";

    buffer_ = new char[sizeof(char) * length_];
    ifs.read(buffer_, length_);
    ifs.close();
  }

  BufferFile() = delete;
  BufferFile(const BufferFile &other) = delete;

  ssize_t GetLength() {
    return length_;
  }
  char* GetBuffer() {
    return buffer_;
  }

  ~BufferFile() {
    delete[] buffer_;
    buffer_ = NULL;
  }
private:
  string file_path_;
  ssize_t length_;
  char* buffer_;
};

/**
 * @brief Constructor for MXNet classifier
 * 
 * @param model_desc Model description json file.
 * @param model_params Model param nd file.
 * @param mean_file Mean image nd file.
 * @param label_file Label list file.
 * @param input_width Width of input data.
 * @param input_height Height of input data.
 */
MXNetClassifier::MXNetClassifier(const string &model_desc,
                                 const string &model_params,
                                 const string &mean_file,
                                 const string &label_file,
                                 const int input_width,
                                 const int input_height)
    : input_geometry_(input_width, input_height),
      num_channels_(3),
      predictor_(nullptr) {

  // Load labels
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line)) {
    labels_.push_back(string(line));
  }

  // Load the model desc and weights
  BufferFile json_data(model_desc.c_str());
  BufferFile param_data(model_params.c_str());

  int dev_type = 1; // GPU
  int dev_id = 0;
  mx_uint num_input_nodes = 1;
  const char* input_key[1] = {"data"};
  const char** input_keys = input_key;

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(3),
                                        static_cast<mx_uint>(input_width),
                                        static_cast<mx_uint>(input_height) };

  MXPredCreate((const char*)json_data.GetBuffer(),
               (const char*)param_data.GetBuffer(),
               static_cast<size_t>(param_data.GetLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               &predictor_);
}

MXNetClassifier::~MXNetClassifier() {
  // Release Predictor
  if (predictor_ != nullptr) {
    MXPredFree(predictor_);
    predictor_ = nullptr;
  }
}

/**
 * @brief Preprocess the input image and fill in the buffer to feed into the network. Preprocess: 
 *  resize, convert to float, normalize
 * 
 * @param image The input image, with any size.
 * @param input_data The pointer to store the processed input, it must have enough capacity.
 */
void MXNetClassifier::Preprocessing(const cv::Mat &image, mx_float *input_data) {
  #define MEAN_VALUE 117.0

  cv::Mat sample_resized;
  if (image.size() != input_geometry_) {
    cv::resize(image, sample_resized, input_geometry_);
  }

  cv::Mat sample_float;
  sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, MEAN_VALUE, sample_normalized);

  std::vector<cv::Mat> split_channels;
  split_channels.resize(num_channels_);
  cv::split(sample_normalized, split_channels);

  mx_float *data = input_data;

  for (int i = 0; i < num_channels_; i++) {
    memcpy(data, split_channels[i].data, input_geometry_.area() * sizeof(float));
    data += input_geometry_.area();
  }
}

std::vector<Prediction> MXNetClassifier::Classify(const cv::Mat &image, int N) {
  size_t image_size = input_geometry_.area() * num_channels_;
  mx_float *input_data = new mx_float[image_size];
  Timer timer, total_timer;
  timer.Start();
  total_timer.Start();
  Preprocessing(image, input_data);
  LOG(INFO) << "Preprocessed in " << timer.ElapsedMSec() << " ms";

  timer.Start();
  MXPredSetInput(predictor_, "data", input_data, image_size);
  MXPredForward(predictor_);
  LOG(INFO) << "Forward in " << timer.ElapsedMSec() << " ms";

  mx_uint output_index = 0;
  mx_uint *output_shape = 0;
  mx_uint output_shape_len;

  // Get Output Result
  timer.Start();
  MXPredGetOutputShape(predictor_, output_index, &output_shape, &output_shape_len);

  size_t output_size = 1;
  for (mx_uint i = 0; i < output_shape_len; ++i)
    output_size *= output_shape[i];

  std::vector<float> output(output_size);

  MXPredGetOutput(predictor_, output_index, &(output[0]), output_size);
  LOG(INFO) << "Get output in " << timer.ElapsedMSec() << " ms";
  std::vector<Prediction> predictions;
  std::vector<int> maxN = Argmax(output, N);

  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  LOG(INFO) << "Total classify done in " << total_timer.ElapsedMSec() << " ms";

  return predictions;
}
