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
import core
from .const import API_PATH

class Pipeline:
    def __init__(self, name, plan):
        """
        Constructor of a pipeline
        :param name: Name of the pipeline
        :param plan: Plan in Streamer Pipeline Language
        """
        self._name = name
        self._plan = plan

    def run(self):
        """
        Run the pipeline.
        :return:
        """
        post_data = {
            "name": self._name,
            "spl": self._plan
        }

        r = core.Core().request(API_PATH["pipelines"], "POST", data=post_data)
        if r['result'] == 'success':
            return True
        else:
            core.LOG.Error("Failed to run pipeline, reason: ", r['reason'])
            return False

    def stop(self):
        """
        Stop the pipeline.
        :return:
        """
        r = core.Core().request(API_PATH["pipeline"], "DELETE", {"pipeline_name": self._name})
        if r['result'] == 'success':
            return True
        else:
            core.LOG.Error("Failed to stop pipeline, reason: ", r['reason'])
            return False

    def processors(self):
        """
        Get the processors inside the pipeline.
        :return: A dictionary of processor_name => processor
        """
        pass

def get_pipelines():
    """
    Get pipelines running on streamer device.
    :return: A list of pipelines
    """
    pass

