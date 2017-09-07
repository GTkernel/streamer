#!/usr/bin/env python3
"""Extracts a segment of video surrounding each keyframe in a video """

import argparse
import math
import os
from os import path
import random
import subprocess
import sys


def __parse_args():
    parser = argparse.ArgumentParser(
        description="Extracts the context around a video's keyframes")
    parser.add_argument("-i", "--input-file", help="The source video file.", required=True)
    parser.add_argument(
        "-x",
        "--input-width",
        help="The width of the input video.",
        required=True,
        type=int)
    parser.add_argument(
        "-y",
        "--input-height",
        help="The height of the input video.",
        required=True,
        type=int)
    parser.add_argument("-m", "--model", help="The name of the model to run.", required=True)
    parser.add_argument("-c", "--config-dir", help="The Streamer config directory.", required=True)
    parser.add_argument("-l", "--layer", help="The intermediate layer use.", required=True)
    parser.add_argument(
        "-s",
        "--sel",
        help="The selectivity to use, in the range (0, 1].",
        required=True,
        type=float)
    parser.add_argument(
        "-b",
        "--buf-len",
        help="The number of frames to buffer before detecting keyframes.",
        required=True,
        type=int)
    parser.add_argument(
        "-o",
        "--output-dir",
        help="The directory in which to store the intermediate results and final videos.",
        required=True)
    parser.add_argument(
        "-r",
        "--streamer-root",
        help="The root directory of a compiled Streamer installation.",
        required=True)
    args = parser.parse_args()

    sel = args.sel
    if sel <= 0 or sel > 1:
        raise Exception("--sel must be in the range (0, 1], but is: {}".format(sel))
    buf_len = args.buf_len
    if not buf_len > 0:
        raise Exception("--buf-len must be greater than 0, but is: {}".format(buf_len))
    return args


def __run(cmd):
    """Runs the provided command and returns its stdout as a string """
    print(cmd)
    # Decode the returned bytes object to a str, then drop the trailing newline.
    return subprocess.run(cmd, shell=True, check=True,
                          stdout=subprocess.PIPE).stdout.decode("UTF-8")[:-1]


def __frames_to_time(frame):
    """Converts a frame count to a time string of the form hours:minutes:seconds """
    fps = 30
    fpm = fps * 60
    fph = fpm * 60

    hrs = math.floor(frame / fph)
    remaining = frame - hrs * fph
    mins = math.floor(frame / fpm)
    remaining -= mins * fpm
    secs = float(remaining) / fps
    return "{}:{}:{}".format(hrs, mins, secs)


def __extract_segment(keyframe, total_frames, segment_dir, input_filepath):
    """Extracts the video segment surrounding the provided keyframe

    Returns the filepath of the video segment.
    """
    dur_f = 60
    start_f = keyframe - 30
    # Adjust the start frame if it is before the start of the video.
    if start_f < 0:
        dur_f += start_f
        start_f = 0
    end_f = start_f + dur_f
    # Adjust the duration if it extends past the end of the video.
    if end_f >= total_frames:
        dur_f = total_frames - start_f

    print("start_f:", start_f)
    start_t = __frames_to_time(start_f)
    print("start_t:", start_t)

    print("dur_f:", dur_f)
    dur_t = __frames_to_time(dur_f)
    print("dur_t:", dur_t)

    segment_filepath = path.join(segment_dir, "{}.mp4".format(keyframe))
    extract_cmd = ("ffmpeg -y -loglevel warning -ss {} -i {} -t {} "
                   "-vcodec copy -acodec copy {}").format(start_t, input_filepath,
                                                          dur_t, segment_filepath)
    __run(extract_cmd)
    return segment_filepath


def main():
    """This program's entrypoint """
    args = __parse_args()

    # Copy the config directory so that we can make changes to it.
    old_config_dir = args.config_dir
    output_dir = args.output_dir
    cp_cmd = "cp -rf {} {}".format(old_config_dir, output_dir)
    __run(cp_cmd)

    # Add the input video as a new Camera and overwrite cameras.toml.
    config_dir = path.join(output_dir, path.basename(path.normpath(old_config_dir)))
    cameras_path = path.join(config_dir, "cameras.toml")
    input_filepath = args.input_file
    camera_name = "keyframe_detector_input_{}".format(random.randrange(sys.maxsize))
    new_camera_toml = ("\n" +
                       "[[camera]]\n" +
                       "name = \"{}\"\n".format(camera_name) +
                       "video_uri = \"file://{}\"\n".format(input_filepath) +
                       "width = {}\n".format(args.input_width) +
                       "height = {}\n".format(args.input_height))
    with open(cameras_path, "w") as cameras_file:
        cameras_file.write(new_camera_toml)

    # Run keyframe_detector app on the raw video, which produces a file containing the indices of
    # the keyframes.
    kd_app = path.join(args.streamer_root, "build", "apps", "keyframe_detector")
    sel = args.sel
    buf_len = args.buf_len
    kd_cmd = ("{} --save-jpegs ".format(kd_app) +
              "--config-dir={} ".format(config_dir) +
              "--camera={} ".format(camera_name) +
              "--model={} ".format(args.model) +
              "--layer={} ".format(args.layer) +
              "--sel={} ".format(sel) +
              "--buf-len={} ".format(buf_len) +
              "--output-dir={}".format(output_dir))
    __run(kd_cmd)

    # Extract frame indices from the keyframe_detector's output log.
    log_filepath = path.join(output_dir, "keyframe_buffer_0_{}_{}.log".format(sel, buf_len))
    if not path.isfile(log_filepath):
        raise Exception("Log file {} does not exist!".format(log_filepath))
    with open(log_filepath, "r") as log_file:
        k_idxs = [int(idx_str) for idx_str in log_file]
    print("Keyframe indices:", k_idxs)

    # Determine the number of frames in the source video.
    frames_cmd = ("ffprobe -v error -count_frames -select_streams v:0 "
                  "-show_entries stream=nb_read_frames "
                  "-of default=nokey=1:noprint_wrappers=1 {}").format(input_filepath)
    total_frames = int(__run(frames_cmd)) - 2
    print("Total frames:", total_frames)

    # Generate video segments surrounding the keyframes.
    segment_dir = path.join(output_dir, "segments")
    if not path.exists(segment_dir):
        os.makedirs(segment_dir)
    segment_filepaths = [__extract_segment(k, total_frames, segment_dir, input_filepath)
                         for k in k_idxs]

    # Create a file containing the paths to all of the segments.
    segment_index_filepath = path.join(output_dir, "segments_index.txt")
    with open(segment_index_filepath, "w") as segment_index_file:
        for segment_filepath in segment_filepaths:
            segment_index_file.write("file '{}'\n".format(segment_filepath))

    # Concatenate the segments to create the final video.
    output_filepath = path.join(output_dir, "output.mp4")
    concat_cmd = "ffmpeg -y -loglevel warning -f concat -safe 0 -i {} -c copy {}".format(
        segment_index_filepath, output_filepath)
    __run(concat_cmd)


if __name__ == "__main__":
    main()
