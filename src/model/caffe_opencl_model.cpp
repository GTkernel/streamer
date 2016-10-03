//
// Created by Ran Xian (xranthoar@gmail.com) on 10/1/16.
//

#include "caffe_opencl_model.h"

template<typename DType>
CaffeOpenCLModel::CaffeOpenCLModel(const ModelDesc &model_desc,
                                   Shape input_shape) : Model(model_desc,
                                                              input_shape) {}
template<typename DType>
void CaffeOpenCLModel::Load() {

}
template<typename DType>
void CaffeOpenCLModel::Evaluate() {

}

template
class CaffeOpenCLModel<float>;
