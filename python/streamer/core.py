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
        url = (self._base_url + path).format(params)
        return url

    def request(self, path, method, params=None, data=None):
        request_url = self._build_url(path, params)
        r = requests.get(request_url)
        return json.loads(r.text)

