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
from .stream import Stream


class Processor:

  def __init__(self, sources, sinks):
    self._sources = sources
    self._sinks = sinks
    self._depend_on = set()
    self._depended_by = set()

  def _set_source(self, name, stream):
    """
    Set a source stream of the processor
    :param name: The name of the stream to set
    :param stream: The stream to be set
    :return: None
    """
    self._sources[name] = stream
    self._depend_on.add(stream.producer)
    stream.producer._depended_by.add(self)

  def _get_sink(self, name):
    """
    Get a stream from the processor
    :param name: The name of the stream to get
    :return: The stream
    """
    return self._sinks[name]

  def to_spl(self):
    """
    Convert the processor to a processor statement
    """
    raise NotImplementedError


class ImageTransformer(Processor):

  def __init__(self, width, height):
    super(Processor, self).__init__({}, {'output': Stream(self)})
    self._width = width
    self._height = height

  def to_spl(self):
    spl = "processor(ImageTransformer, width={width}, height={height})".format(
        width=self._width, height=self._height)

    return spl


class ImageClassifier(Processor):

  def __init__(self, network, width, height):
    super(Processor, self).__init__({}, {'output': Stream(self)})
    self._network = network
    self._width = width
    self._height = height

  def to_spl(self):
    spl = "processor(ImageClassifier, network={self.network}, width={self.width}, height={self.height}".format(
        self=self)

    return spl


class DummyNNProcessor(Processor):
  pass


class OpenCVFaceDetector(Processor):
  pass


class VideoEncoder(Processor):
  pass
