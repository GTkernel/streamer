// Copyright 2016 The Streamer Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Created by Ran Xian (xranthoar@gmail.com) on 9/24/16.
//

#include "model.h"
#include <string>
#include <vector>
#include "common/types.h"

Model::Model(const ModelDesc& model_desc, Shape input_shape, size_t batch_size)
    : model_desc_(model_desc),
      input_shape_(input_shape),
      batch_size_(batch_size) {}

Model::~Model() {}

ModelDesc Model::GetModelDesc() const { return model_desc_; }

cv::Mat Model::ConvertAndNormalize(cv::Mat img) { return img; }
