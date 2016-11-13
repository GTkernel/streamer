import core
import config
from .const import API_PATH
from .pipeline import Pipeline

import subprocess
import time
from multiprocessing import Process


class CameraFile:
    def __init__(self, filename, size, created_time):
        self.filename = filename
        self.size = size
        self.created_time = created_time
        pass

    def download(self, location):
        """
        Download a file from a streamer device. The method will block until the
        file is downloaded.
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
        self.started = True if attributes['started'] == 'true' else False

    def __str__(self):
        return "[Camera:{self.name}]: " \
               "{self.width}x{self.height}, {self.video_uri}, {self.started}".format(self=self)

    def capture(self, filename="img.jpeg"):
        """
        Capture an image from the camera, and save to a file.
        :param filename: The name of the file to save the image.
        """
        image_bytes = core.Core().request(API_PATH['capture'],
                                          'GET',
                                          params={'camera_name': self.name},
                                          load_json=False)
        with open(filename, "wb") as f:
            f.write(image_bytes)

    def stream(self):
        """
        Stream the video of the camera through RTSP.
        :return: The RTSP URI of the streamed video.
        """
        pass

    def _stream_udp(self, port):
        decoder = self._get_decoder()
        GST_PIPELINE = \
        """
        gst-launch-1.0 -v udpsrc address={host} port={port} ! application/x-rtp ! rtph264depay ! {decoder} ! videoconvert ! autovideosink sync=false
        """.format(host=config.STREAMER_SERVER_HOST, port=port, decoder=decoder)
        print GST_PIPELINE
        subprocess.call(GST_PIPELINE, shell=True)

    def preview(self, width=None, height=None):
        """
        Preview the video of the camera, the method will not return anything but display the camera directly.
        :param width: Width of the video. Leaving it None will use the width of the original camera stream.
        :param height: Height of the video. Leaving it None will use the height of the original camera stream.
        :return: Nothing
        """

        # Start a pipeline in another thread
        import random
        # Get a random port
        # FIXME: this is not guaranteed to use a free port on the server, should
        # have a more robust way.
        random_port = random.randrange(10000, 30000)

        STREAM_VIDEO_SPL = \
        """
        camera = camera({name})
        video_encoder = processor(VideoEncoder, port={port}, width={width}, height={height})
        video_encoder[input] = camera[bgr_output]
        """.format(name=self.name, port=random_port, width=self.width, height=self.height)
        pipeline = Pipeline("stream_{}_{}".format(self.name, random_port),
                            STREAM_VIDEO_SPL)
        r = pipeline.run()

        if not r:
            return

        self._stream_udp(random_port)

        pipeline.stop()

    def record(self, duration, compress=True):
        """
        Record the camera video for a given duration.
        :param duration: the duration (in s) of the video to record.
        :param compress: compress the video or not.
        :return: A filename of the recorded video. The filename might not be used to retrieve the video until the camera has actually finished recording.
        """
        from time import strftime, localtime
        record_time = strftime("%Y-%m-%d+%H:%M:%S", localtime())
        filename = self.name + "/" + record_time + ".mp4"
        if compress:
            RECORD_VIDEO_SPL = \
            """
            camera = camera({name})
            video_encoder = processor(VideoEncoder, filename={filename}, width={width}, height={height})

            video_encoder[input] = camera[bgr_output]
            """.format(name=self.name, filename=filename, width=self.width, height=self.height)
        else:
            RECORD_VIDEO_SPL = \
            """
            camera = camera({name})
            file_writer = processor(FileWriter, filename={filename})

            file_writer[input] = camera[raw_output]
            """.format(name=self.name, filename=filename)

        # FIXME: this assumes that there is only one record pipeline on any
        # given camera.
        pipeline = Pipeline("record_{}".format(self.name), RECORD_VIDEO_SPL)

        r = pipeline.run()

        if not r:
            return

        time.sleep(duration)
        pipeline.stop()

    def files(self):
        """
        Get the list of files generated from the camera.
        :return: A list of files
        """
        

    def control(self, params):
        """
        Control the camera parameters.
        :param params: The settings of the cameras.
        :return: True if the configured succeeded, False otherwise. On False, an error message will be printed.
        """
        pass

    def _get_decoder(self):
        import platform
        system = platform.system()
        if system == "Darwin":
            return "avdec_h264"
        return "x264dec"


def get_cameras():
    r = core.Core().request(API_PATH['cameras'], 'GET')
    cameras = []
    for camera in r['cameras']:
        cameras.append(Camera(camera))
    return cameras
