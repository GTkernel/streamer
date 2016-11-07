import core

class Video:
    def __init__(self, filename, size, created_time):
        self.filename = filename
        self.size = size
        self.created_time = created_time
        pass

    def download(self, location):
        """
        Download a video from a streamer device. The method will block until video is downloaded.
        :param location: Location to store the video.
        :return:
        """
        pass

class Camera:
    def __init__(self, attributes):
        self.name = attributes['name']
        self.height = int(attributes['height'])
        self.width = int(attributes['width'])
        self.video_uri = attributes['video_uri']

    def __str__(self):
        return "[Camera:{self.name}]: " \
               "{self.width}x{self.height}, {self.video_uri}".format(self=self)

    def capture(self, filename="img.jpeg"):
        """
        Capture an image from the camera, and save to a file.
        :param filename: The name of the file to save the image.
        """
        pass

    def stream(self):
        """
        Stream the video of the camera through RTSP.
        :return: The RTSP URI of the streamed video.
        """
        pass

    def preview(self, width=None, height=None):
        """
        Preview the video of the camera, the method will not return anything but display the camera directly.
        :param width: Width of the video. Leaving it None will use the width of the original camera stream.
        :param height: Height of the video. Leaving it None will use the height of the original camera stream.
        :return: Nothing
        """
        pass

    def record(self, duration):
        """
        Record the camera video for a given duration.
        :param duration: the duration (in s) of the video to record.
        :return: A filename of the recorded video. The filename might not be used to retrieve the video
                 until the camera has actually finished recording.
        """
        pass

    def videos(self):
        """
        Get the list of recorded videos on the camera.
        :return: A list of videos, each video
        """

    def control(self, params):
        """
        Control the camera parameters.
        :param params: The settings of the cameras.
        :return: True if the configured succeeded, False otherwise. On False, an error message will be printed.
        """
        pass

def get_cameras():
    r = core.Core().request("/cameras", "GET")
    cameras = []
    for camera in r['cameras']:
        cameras.append(Camera(camera))
    return cameras