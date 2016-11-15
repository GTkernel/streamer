"""Streamer Python API consts"""

# All API paths
API_PATH = {
    'cameras': '/cameras',
    'camera': '/cameras/{camera_name}',
    'camera_files': '/cameras/{camera_name}/files',
    'capture': '/cameras/{camera_name}/capture',
    'camera_control': '/cameras/{camera_name}/control',
    'pipelines': '/pipelines',
    'pipeline': '/pipelines/{pipeline_name}',
    'processors': '/processors',
    'processor': '/processors/{processor_id}',
    'streams': '/streams',
    'stream': '/streams/{stream_id}',
    'download': '/download'
}
