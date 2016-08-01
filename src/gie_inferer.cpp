//
// Created by Ran Xian on 7/26/16.
//

#include "gie_inferer.h"
#include <cuda_runtime_api.h>

template<typename DType>
gie_inferer<DType>::gie_inferer(const string &deploy_file, const string &model_file, const string &input_blob_name, const string &output_blob_name):
    deploy_file_(deploy_file),
    model_file_(model_file),
    input_blob_name_(input_blob_name),
    output_blob_name_(output_blob_name),
    infer_runtime_(nullptr),
    engine_(nullptr),
    d_input_buffer(nullptr),
    d_output_buffer(nullptr) {
  if (sizeof(DType) == 16) {
    IBuilder *builder = createInferBuilder(logger_);
    bool supportFp16 = builder->plaformHasFastFp16();
    builder->destroy();
    CHECK(supportFp16) << "Platform does not support fp16 but gie_inferer is initialized with fp16";
  }
}

/**
 * \brief Logger for GIE.
 * \param severity
 * \param msg
 */
template<typename DType>
void gie_inferer<DType>::Logger::log(ILogger::Severity severity,
                                                    const char *msg) {
  switch (severity) {
    case Severity::kERROR:
      LOG(ERROR) << "GIE: " << msg;
      break;
    case Severity::kINFO:
      LOG(INFO) << "GIE: " << msg;
      break;
    case Severity::kINTERNAL_ERROR:
      LOG(ERROR) << "GIE internal: " << msg;
      break;
    case Severity::kWARNING:
      LOG(WARNING) << "GIE: " << msg;
      break;
  }
}

/**
 * \brief Transform Caffe model to GIE model.
 * \param deploy_file Caffe .proto file.
 * \param model_file Caffe .caffemodel file.
 * \param outputs Network outputs.
 * \param max_batch_size Maximum batch size.
 * \param gie_model_stream The stream to GIE model.
 */
template<typename DType>
void gie_inferer<DType>::CaffeToGIEModel(const string &deploy_file,
                                 const string &model_file,
                                 const std::vector<string> &outputs,
                                 unsigned int max_batch_size,
                                 std::ostream &gie_model_stream) {
  // Create API root class - must span the lifetime of the engine usage.
  IBuilder *builder = createInferBuilder(logger_);
  INetworkDefinition *network = builder->createNetwork();

  // Parse the caffe model to populate the network, then set the outputs
  std::shared_ptr<CaffeParser> parser(new CaffeParser);

  // Determine data type
  bool useFp16 = builder->plaformHasFastFp16();
  LOG(INFO) << "GIE use FP16: " << (useFp16 ? "YES" : "NO");

  // Get blob:tensor name mappings
  DataType model_data_type = useFp16 ? DataType::kHALF : DataType::kFLOAT;
  const IBlobNameToTensor *blob_name_to_tensor = parser->parse(deploy_file.c_str(),
                                                            model_file.c_str(),
                                                            *network,
                                                            model_data_type);
  CHECK(blob_name_to_tensor != nullptr) << "Map from blob name to tensor is null";

  // Mark outputs
  for (auto &s : outputs) {
    network->markOutput(*blob_name_to_tensor->find(s.c_str()));
  }

  // Build the engine
  builder->setMaxBatchSize(max_batch_size);
  builder->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);

  // Set up the network for paired-fp16 format.
  if (useFp16)
    builder->setHalf2Mode(true);

  ICudaEngine *engine = builder->buildCudaEngine(*network);
  CHECK(engine != nullptr) << "GIE can't build engine";

  // We don't need the network any more, and we can destroy the parser
  network->destroy();

  // Serialize the engine, then close everything down
  engine->serialize(gie_model_stream);
  engine->destroy();
  builder->destroy();
}

template<typename DType>
void gie_inferer<DType>::CreateEngine() {
  gie_model_stream_.seekg(0, gie_model_stream_.beg);

  CaffeToGIEModel(deploy_file_, model_file_, {output_blob_name_}, BATCH_SIZE, gie_model_stream_);

  // Create an engine
  gie_model_stream_.seekg(0, gie_model_stream_.beg);
  infer_runtime_ = createInferRuntime(logger_);
  engine_ = infer_runtime_->deserializeCudaEngine(gie_model_stream_);

  // Get input information
  int input_index = engine_->getBindingIndex(input_blob_name_.c_str());
  int output_index = engine_->getBindingIndex(output_blob_name_.c_str());

  Dims3 input_dims = engine_->getBindingDimensions(input_index);
  Dims3 output_dims = engine_->getBindingDimensions(output_index);

  input_shape_ = Shape(input_dims.c, input_dims.w, input_dims.h);
  output_shape_ = Shape(output_dims.c, output_dims.w, output_dims.h);

  input_size_ = BATCH_SIZE * input_shape_.GetVolume() * sizeof(DType);
  output_size_ = BATCH_SIZE * output_shape_.GetVolume() * sizeof(DType);

  CHECK_EQ(cudaMalloc((void **)(&d_input_buffer), input_size_), cudaSuccess) << "Can't malloc device input buffer";
  CHECK_EQ(cudaMalloc((void **)(&d_output_buffer), output_size_), cudaSuccess) << "Can't malloc device output buffer";
}

template<typename DType>
void gie_inferer<DType>::DoInference(DType *input, DType *output) {
  IExecutionContext *context = engine_->createExecutionContext();
  CHECK(context != nullptr) << "GIE error, can't create context";
  CHECK(input != nullptr) << "Input is invalid: nullptr";
  CHECK(output != nullptr) << "Output is invalid, nullptr";
  CHECK(d_input_buffer != nullptr) << "Device input buffer is not allocated";
  CHECK(d_output_buffer != nullptr) << "Device output buffer is not allocated";
  CHECK(engine_->getNbBindings() == 2);

  void *buffers[2];

  int inputIndex = engine_->getBindingIndex(input_blob_name_.c_str()),
      outputIndex = engine_->getBindingIndex(output_blob_name_.c_str());

  CHECK_EQ(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * input_size_), cudaSuccess);
  CHECK_EQ(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * output_size_), cudaSuccess);

  cudaStream_t stream;
  CHECK_EQ(cudaStreamCreate(&stream), cudaSuccess) << "CUDA error, can't create cuda stream";

  // DMA the input to the GPU, execute the batch asynchronously, and DMA it back
  CHECK_EQ(cudaMemcpyAsync(buffers[inputIndex], input, input_size_ * BATCH_SIZE, cudaMemcpyHostToDevice, stream), cudaSuccess) << "CUDA error, can't async memcpy input to device";
  context->enqueue(BATCH_SIZE, buffers, stream, nullptr);
  CHECK_EQ(cudaMemcpyAsync(output, buffers[outputIndex], output_size_ * BATCH_SIZE, cudaMemcpyDeviceToHost, stream), cudaSuccess) << "CUDA error, can't async memcpy to output from device";
  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);
  CHECK_EQ(cudaFree(buffers[inputIndex]), cudaSuccess);
  CHECK_EQ(cudaFree(buffers[outputIndex]), cudaSuccess);
  context->destroy();
}

template<typename DType>
void gie_inferer<DType>::DestroyEngine() {
  engine_->destroy();
  engine_ = nullptr;
  infer_runtime_->destroy();
  infer_runtime_ = nullptr;
  CHECK_EQ(cudaFree(d_input_buffer), cudaSuccess) << "Can't free device input buffer";
  CHECK_EQ(cudaFree(d_output_buffer), cudaSuccess) << "Can't free device output buffer";
}

template<typename DType>
Shape gie_inferer<DType>::GetInputShape() {
  return input_shape_;
}

template<typename DType>
Shape gie_inferer<DType>::GetOutputShape() {
  return output_shape_;
}

template class gie_inferer<float>;
#ifdef ON_TEGRA
template class GIEInferer<float16>;
#endif
