class Pipeline:
    def __init__(self, name, plan):
        """
        Constructor of a pipeline
        :param name: Name of the pipeline
        :param plan: Plan in Streamer Pipeline Language
        """
        self._name = name
        self._plan = plan
        pass

    def run(self):
        """
        Run the pipeline.
        :return:
        """
        pass

    def stop(self):
        """
        Stop the pipeline.
        :return:
        """
        pass

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

