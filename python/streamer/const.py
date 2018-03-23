# Copyright 2016 The Streamer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streamer Python API consts"""

# All API paths
API_PATH = {
    'cameras': '/cameras',
    'camera': '/cameras/{camera_name}',
    'camera_files': '/cameras/{camera_name}/files',
    'capture': '/cameras/{camera_name}/capture',
    'camera_control': '/cameras/{camera_name}/control',
    'pipelines': '/pipelines',
    'pipeline': '/pipelines/{pipeline_name}',
    'processors': '/processors',
    'processor': '/processors/{processor_id}',
    'streams': '/streams',
    'stream': '/streams/{stream_id}',
    'download': '/download'
}
