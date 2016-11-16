class Stream:

  def __init__(self, producer):
    self.producer = producer

  def connect(self):
    """
    Connect the stream with the zmq topic
    """

  def pop_frame(self):
    """
    Pop a frame from the stream.
    :return: A frame in the front of the stream queue.
    """
    pass

  def push_stream(self):
    """
    Push a frame into the stream.
    :return:
    """
    # TODO: need more design on this.
    pass
