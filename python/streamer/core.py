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
"""Core communication functions between streamer server and the python wrapper"""

import logging
import config
import requests
import json

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s: %(message)s')
LOG = logging.getLogger('streamer')

class Core:
    def __init__(self):
        self._host = config.STREAMER_SERVER_HOST
        self._port = config.STREAMER_SERVER_PORT
        self._base_url = "http://{}:{}".format(self._host, self._port)

    def _build_url(self, path, params):
        if params != None:
            url = (self._base_url + path).format(**params)
        else:
            url = (self._base_url + path).format(params)
        return url

    def request(self, path, method, params=None, data=None, load_json=True, stream=False):
        request_url = self._build_url(path, params)

        if method == 'GET':
            r = requests.get(request_url, json=data, stream=stream)
        elif method == 'POST':
            r = requests.post(request_url, json=data, stream=stream)
        elif method == 'DELETE':
            r = requests.delete(request_url, json=data, stream=stream)

        # Json data
        if load_json:
            return json.loads(r.text)
        else:
            # Binary data
            return r

