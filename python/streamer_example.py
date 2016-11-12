"""The Wrapper Class for Streamer's REST API"""
import streamer.camera


## List all cameras
cameras = streamer.camera.get_cameras()

for camera in cameras:
    print camera

# Capture an image from one of the camera
camera = cameras[0]
camera.capture(camera.name + ".jpeg")
