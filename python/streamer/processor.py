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
