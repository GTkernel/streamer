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

    def request(self, path, method, params=None, data=None, load_json=True):
        request_url = self._build_url(path, params)

        if method == 'GET':
            r = requests.get(request_url, json=data)
        elif method == 'POST':
            r = requests.post(request_url, json=data)
        elif method == 'DELETE':
            r = requests.delete(request_url, json=data)

        # Json data
        if load_json:
            return json.loads(r.text)
        else:
            # Binary data
            return r.content

