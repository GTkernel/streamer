#!/usr/bin/env python3

import math
import os
import subprocess
import cv2
import argparse

def __parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Keyframe detector.");
    parser.add_argument("--input", dest="input_filepath",
            help="Input file name",
            required=True)
    parser.add_argument("--outdir", dest="output_dirname", default=".",
            help="Output directory (default: .)")
    parser.add_argument("--extract", dest="extract_segment",
            action="store_true", default=False,
            help="Extract a segment of the input video")
    parser.add_argument("--start_frame", dest="start_frame", default=0,
            help="Start frame for extraction (default: 0)",
            type=int)
    parser.add_argument("--num_frames", dest="num_frames", default=1200,
            help="Number of frames to extract (default: 1200)",
            type=int)
    args = parser.parse_args()
    return args

def __run(cmd):
    """Runs the provided command and returns its stdout as a string """
    print(cmd)
    # Decode the returned bytes object to a str, then drop the trailing newline.
    return subprocess.run(cmd, shell=True, check=True,
                          stdout=subprocess.PIPE).stdout.decode("UTF-8")[:-1]

def __get_video_properties(path):
    """Returns the properties of input video (dimensions, frame rate, and number of frames"""
    vid = cv2.VideoCapture(path)
    if vid.isOpened():
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print("{}: width({}), height({}), fps({}), num_frames({})".format(
            os.path.basename(path), width, height, fps, num_frames))
    return width, height, fps, num_frames

def __frames_to_time(frame, fps):
    """Converts a frame number to a time string of the form hours:minutes:seconds """
    fpm = fps * 60
    fph = fpm * 60

    hrs = math.floor(frame / fph)
    remaining = frame - hrs * fph
    mins = math.floor(frame / fpm)
    remaining -= mins * fpm
    secs = float(remaining) / fps
    return "{}:{}:{}".format(int(hrs), int(mins), int(secs))

def __align_frame(frame, fps):
    """Aligns the frame number to the second boundary"""
    return ((frame + (fps - 1)) // fps) * fps

def __extract_segment(input_filepath, fps, start_frame, num_frames, output_dirname="."):
    """Extracts the video segment from start_f (for dur_f)
    Returns the filepath of the video segment.
    """
    start_f = __align_frame(start_frame, fps)
    dur_f = __align_frame(num_frames, fps)
    start_t = __frames_to_time(start_f, fps)
    dur_t = __frames_to_time(dur_f, fps)

    print("start_f: {}, dur_f: {}".format(start_f, dur_f))
    print("start_t: {}, dur_t: {}".format(start_t, dur_t))

    output_filename = "{}-s{}-d{}.mp4".format(
            os.path.splitext(os.path.basename(input_filepath))[0],
            int(start_f), int(dur_f))
    output_filepath = os.path.join(output_dirname, output_filename)

    print("Writing video segment to {}".format(output_filepath))
    extract_cmd = ("ffmpeg -y -loglevel warning -ss {} -i {} -t {} "
                   "-vcodec copy -acodec copy {}").format(start_t,
                           input_filepath, dur_t, output_filepath)
    __run(extract_cmd)
    return output_filepath


def main():
    args = __parse_args()
    input_filepath = args.input_filepath

    (width, height, fps, num_frames) = __get_video_properties(input_filepath)
    if args.extract_segment:
        __extract_segment(input_filepath, fps, args.start_frame, args.num_frames, args.output_dirname)


if __name__ == "__main__":
    main()
