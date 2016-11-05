class Processor:
    def __init__(self, sources, sinks):
        self._sources = sources
        self._sinks = sinks
        pass

    def set_source(self, name, stream):
        """
        Set a source stream of the processor
        :param name: The name of the stream to set
        :param stream: The stream to be set
        :return: None
        """
        pass

    def get_sink(self, name):
        """
        Get a stream from the processor
        :param name: The name of the stream to get
        :return: The stream
        """
        pass