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
"""The Wrapper Class for Streamer's REST API"""
import streamer.camera
import streamer.pipeline


# List all cameras
cameras = streamer.camera.get_cameras()

for camera in cameras:
  print camera

# Capture an image from one of the camera
print "Capture image..."
camera = cameras[0]
camera.capture(camera.name + ".jpeg")

# Control the camera, and take a picture again
print "Pan/Tile..."
camera.control({'pan': 'left'})

# Stream and preview the video of the camera
print "Streaming..."
camera.preview(640, 480)

# Record 10 seconds of the camera with compression
print "Recording 10 secs..."
camera.record(10, compress=True)

# Get a list of recorded videos of a camera
files = camera.files()
for file in files:
  print file

# Download the file
  files[1].download()

# Compose and run a streamer pipeline

transformer = streamer.processor.ImageTransformer(227, 227)
classifier = streamer.processor.ImageClassifier('GoogleNet', 227, 227)

transformer['input'] = camera['output']
classifier['input'] = transformer['output']

pipeline = streamer.pipeline.Pipeline('classification')
pipeline.add('camera', camera)
pipeline.add('transformer', transformer)
pipeline.add('classifier', classifier)

pipeline.run()
labels_stream = pipeline['classifier']['labels']
labels_stream.connect()
for i in xrange(30):
  frame = labels_stream.pop_frame()
  print frame
labels_stream.disconnect()
pipeline.stop()

# Should be the same as:
# image_classification_pipeline = """
# camera = camera({camera_name})
# classifier = processor(ImageClassifier, network=GoogleNet, width=227, height=227)
# classifier[labels].publish({topic})
# """.format(camera_name=camera.name, topic=topic_name)

# pipeline = streamer.Pipeline(image_classification_pipeline)
# pipeline.start()

topic = Topic(topic_name)

for event in topic.events:
  print event
