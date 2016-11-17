import core
import config
from .const import API_PATH
from .pipeline import Pipeline

import subprocess
import time
from multiprocessing import Process


class CameraFile:

  def __init__(self, path, size, created_time):
    self.path = path
    self.size = size
    self.created_time = created_time

  def download(self, location=None):
    """
    Download a file from a streamer device. The method will block until the
    file is downloaded.
    :param location: Location to store the video.
    :return: The path to the downloaded file.
    """
    r = core.Core().request(API_PATH['download'], 'POST',
                            data={'path': self.path}, stream=True, load_json=False)
    if location == None:
      local_filename = self.path.split("/")[-1]
    else:
      local_filename = location
    with open(local_filename, 'wb') as f:
      for chunk in r.iter_content(chunk_size=1024):
        if chunk:
          f.write(chunk)

    return local_filename

  def __str__(self):
    size = float(self.size)
    unit = "B"
    if size > 1024:
      size /= 1024
      unit = "KB"

    if size > 1024:
      size /= 1024
      unit = "MB"

    if size > 1024:
      size /= 1024
      unit = "GB"

    size_str = "%.2f %s" % (size, unit)
    return "[File:{self.path}]: " \
           "{size_str}, {self.created_time}".format(
               self=self, size_str=size_str)


class Camera:

  def __init__(self, attributes):
    self.name = attributes['name']
    self.height = int(attributes['height'])
    self.width = int(attributes['width'])
    self.video_uri = attributes['video_uri']
    self.started = True if attributes['started'] == 'true' else False

  def __str__(self):
    return "[Camera:{self.name}]: " \
           "{self.width}x{self.height}, {self.video_uri}, {self.started}".format(
               self=self)

  def capture(self, filename="img.jpeg"):
    """
    Capture an image from the camera, and save to a file.
    :param filename: The name of the file to save the image.
    """
    r = core.Core().request(API_PATH['capture'],
                            'GET',
                            params={'camera_name': self.name},
                            load_json=False)
    with open(filename, "wb") as f:
      f.write(r.content)

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
        gst-launch-1.0 udpsrc address={host} port={port} ! application/x-rtp ! rtph264depay ! {decoder} ! videoconvert ! autovideosink sync=false
        """.format(host=config.STREAMER_SERVER_HOST, port=port, decoder=decoder)
    print GST_PIPELINE
    subprocess.call(GST_PIPELINE, shell=True)

  def _stream_tcp(self, port):
    decoder = self._get_decoder()
    GST_PIPELINE = \
        """
        gst-launch-1.0 tcpclientsrc port={port} host={host} ! tsdemux ! h264parse ! {decoder} ! videoconvert ! autovideosink sync=false
        """.format(host=config.STREAMER_SERVER_HOST, port=port, decoder=decoder)
    print GST_PIPELINE
    subprocess.call(GST_PIPELINE, shell=True)

  def preview(self, width, height, tcp=True):
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
    # Fix it to 12345 for debugging with ssh tunnel
    random_port = 12345

    STREAM_VIDEO_SPL = \
        """
        camera = camera({name})
        video_encoder = processor(VideoEncoder, port={port}, width={width}, height={height})
        video_encoder[input] = camera[bgr_output]
        """.format(name=self.name, port=random_port, width=width, height=height)
    pipeline = Pipeline("stream_{}_{}".format(self.name, random_port),
                        STREAM_VIDEO_SPL)
    r = pipeline.run()

    if not r:
      return

    if tcp:
      self._stream_tcp(random_port)
    else:
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
    filename = self.name + "/" + record_time
    if compress:
      filename += ".mp4"
      RECORD_VIDEO_SPL = \
          """
            camera = camera({name})
            video_encoder = processor(VideoEncoder, filename={filename}, width={width}, height={height})

            video_encoder[input] = camera[bgr_output]
            """.format(name=self.name, filename=filename, width=self.width, height=self.height)
    else:
      filename += ".dat"
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
    r = core.Core().request(API_PATH['camera_files'], 'GET',
                            params={'camera_name': self.name})
    files = []
    for file_json in r['files']:
      file = CameraFile(file_json['path'], int(file_json['size']),
                        file_json['created_at'])
      files.append(file)
    return files

  def control(self, configs):
    """
    Control the camera parameters.
    :param params: The settings of the cameras.
    :return: True if the configured succeeded, False otherwise. On False, an error message will be printed.
    """
    r = core.Core().request(API_PATH['camera_control'],
                            'POST', params={'camera_name': self.name}, data=configs)

    if r['result'] == 'success':
      return True
    else:
      return False

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
