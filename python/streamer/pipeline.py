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

