#!/usr/bin/env python3
""" Varies either the selectivity, buffer lenght, or numnber of levels """

import argparse
import os
from os import path
import subprocess

import numpy


def validate_file(pred, filepath, param_name):
    """Throws an error if filepath is None or does not exist """
    if pred:
        if filepath is None:
            raise Exception("\"" + param_name + "\" is required!")
        if not path.isfile(filepath):
            raise Exception("\"" + filepath + "\" does not exist!")


def __validate_sel(sel):
    if sel <= 0 or sel > 1:
        raise Exception("--sel must be in the range (0, 1], but is: {}".format(sel))


def __validate_buf_len(buf_len):
    if not buf_len > 0:
        raise Exception("--buf-len must be greater than 0, but is: {}".format(buf_len))


def __validate_levels(levels):
    if not levels > 0:
        raise Exception("--levels must be greater than 0, but is: {}".format(levels))


def __parse_config_file(filepath, dest_type):
    if filepath is not None:
        return [dest_type(line[:-1]) for line in open(filepath, "r") if line[0] != "#"]
    else:
        return None


def parse_args(parser, validate_func):
    """Returns an object containing the arguments to this program """
    if parser is None:
        parser = argparse.ArgumentParser(
            description=("Generates a graph of the duration of the keyframe detection algorithm "
                         "when varying a particular parameter."))
    parser.add_argument(
        "-s",
        "--sel",
        help="The selectivity to use, in the range (0, 1].",
        required=True,
        type=float)
    parser.add_argument(
        "--vary-sel",
        action="store_true",
        help="Vary the keyframe detector's selectivity.")
    parser.add_argument(
        "--sels-file",
        help="A file containing the selectivities to use, in the range (0, 1].",
        required=False)
    parser.add_argument(
        "-b",
        "--buf-len",
        help="The number of frames to buffer before detecting keyframes.",
        required=True,
        type=int)
    parser.add_argument(
        "--vary-buf-len",
        action="store_true",
        help="Vary the keyframe detector's buffer length.")
    parser.add_argument(
        "--buf-lens-file",
        help="A file containing the numbers of frames to buffer before detecting keyframes.",
        required=False)
    parser.add_argument(
        "-v",
        "--levels",
        help="The number of levels in the keyframe detector hierarchy.",
        required=True,
        type=int)
    parser.add_argument(
        "--vary-levels",
        action="store_true",
        help="Vary the keyframe detector's levels.")
    parser.add_argument(
        "--levels-file",
        help="A file containing the numbers of levels in the keyframe detector hierarchy.",
        required=False)
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
    parser.add_argument(
        "-c",
        "--config-dir",
        help="Streamer's configuration directory.",
        required=True)
    parser.add_argument(
        "--queue-size",
        default=16,
        help="The queue size between processors.",
        required=False,
        type=int)
    parser.add_argument(
        "--block",
        action="store_true",
        help="Whether to block when pushing frames.")
    parser.add_argument(
        "-w",
        "--warmup",
        default=5,
        help="The number of warmup trials to run",
        required=False,
        type=int)
    parser.add_argument(
        "-t",
        "--trials",
        default=100,
        help="The number of trials to run.",
        required=False,
        type=int)
    parser.add_argument(
        "-l",
        "--vishash-length",
        default=1024,
        help="The length of the fake vishashes.",
        required=False,
        type=int)

    args = parser.parse_args()

    # Parse selectivities
    __validate_sel(args.sel)
    validate_file(args.vary_sel, args.sels_file, "--sels-file")
    args.sels = __parse_config_file(args.sels_file, float)

    # Parse buffer lengths
    __validate_buf_len(args.buf_len)
    validate_file(args.vary_buf_len, args.buf_lens_file, "--buf-lens-file")
    args.buf_lens = __parse_config_file(args.buf_lens_file, int)

    # Parse levels
    __validate_levels(args.levels)
    validate_file(args.vary_levels, args.levels_file, "--levels-file")
    args.nums_levels = __parse_config_file(args.levels_file, int)

    if not (args.vary_sel or args.vary_buf_len or args.vary_levels):
        raise Exception("Must specify either \"--vary-sel\", \"--vary-buf-len\", or "
                        "\"--vary-levels\"!")

    if validate_func is not None:
        validate_func(args)

    queue_size = args.queue_size
    if queue_size < 0:
        raise Exception("\"--queue-size\" cannot be negative, but is: {}".format(queue_size))

    return args


def __run(cmd):
    """Runs the provided command and returns its stdout as a string """
    print(cmd)
    # Decode the returned bytes object to a str, then drop the trailing newline.
    with open(os.devnull, "w") as devnull:
        return subprocess.run(cmd, shell=True, check=True, stderr=devnull,
                              stdout=subprocess.PIPE).stdout.decode("UTF-8")[:-1]


def __run_kd(streamer_root, config_dir, queue_size, block, num_frames, fake_vishash_length, sel,
             buf_len, levels, output_dir):
    kd_app = path.join(streamer_root, "build", "apps", "keyframe_detector", "vary_kd_params")
    kd_cmd = ("{} ".format(kd_app) +
              "--config-dir {} ".format(config_dir) +
              "--queue-size {} ".format(queue_size) +
              "{}".format("--block " if block else "") +
              "--num-frames {} ".format(int(num_frames)) +
              "--fake-vishashes " +
              "--fake-vishash-length {} ".format(fake_vishash_length) +
              "--sels {} ".format(sel) +
              "--buf-lens {} ".format(buf_len) +
              "--levels {} ".format(levels) +
              "--output-dir {}".format(output_dir))
    __run(kd_cmd)

def __num_frames_to_trigger_detection(sel, buf_len, levels):
    """This only works if the sel * buf_len of each buffer evenly divides the next buffer """
    return ((1.0 / sel) ** (levels - 1)) * buf_len

def __vary(streamer_root, config_dir, queue_size, block, fake_vishash_length, num_trials, vary_sel,
           vary_buf_len, vary_levels, sel, sels, buf_len, buf_lens, levels, nums_levels,
           master_output_dir):
    dir_prefix = ""
    values = None
    if vary_sel:
        dir_prefix = "sel"
        values = sels
    elif vary_buf_len:
        dir_prefix = "buf_lens"
        values = buf_lens
    elif vary_levels:
        dir_prefix = "levels"
        values = nums_levels
    else:
        raise Exception("Must vary something!")

    micros_files = []
    for value in values:
        output_dir = path.join(master_output_dir, dir_prefix + "_" + str(value))

        if not path.isdir(output_dir):
            os.mkdir(output_dir)

        true_sel = sel
        true_buf_len = buf_len
        true_levels = levels

        if vary_sel:
            true_sel = value
        elif vary_buf_len:
            true_buf_len = value
        elif vary_levels:
            true_levels = value
        else:
            raise Exception("Must vary something!")

        num_frames = num_trials * __num_frames_to_trigger_detection(
            true_sel, true_buf_len, true_levels)
        __run_kd(streamer_root, config_dir, queue_size, block, num_frames, fake_vishash_length,
                 true_sel, true_buf_len, true_levels, output_dir)
        micros_file = path.join(output_dir,
                                "{}_{}_{}".format(true_sel, true_buf_len, true_levels),
                                "keyframe_detector_{}_{}_{}_micros.txt".format(
                                    true_sel, true_buf_len, true_levels))
        micros_files.append((value, micros_file))

    return micros_files


def get_key(vary_sel, vary_buf_len, vary_levels):
    """Returns the string name of the parameter that is being varied """
    if vary_sel:
        return "sel"
    elif vary_buf_len:
        return "buf_len"
    elif vary_levels:
        return "levels"
    else:
        raise Exception("Must vary something!")

def run(args, filename_suffix):
    """Runs the keyframe detector, varying a parameter according to args """
    vary_sel = args.vary_sel
    vary_buf_len = args.vary_buf_len
    vary_levels = args.vary_levels
    output_dir = args.output_dir

    # Run the experiments.
    num_warmup_trials = args.warmup
    num_trials = num_warmup_trials + args.trials
    print("num_trials: {}".format(num_trials))
    micros_files = __vary(args.streamer_root, args.config_dir, args.queue_size, args.block,
                          args.vishash_length, num_trials, vary_sel, vary_buf_len, vary_levels,
                          args.sel, args.sels, args.buf_len, args.buf_lens, args.levels,
                          args.nums_levels, output_dir)

    # Build output CSV file for graphing script.
    key = get_key(vary_sel, vary_buf_len, vary_levels)
    suffix_to_use = "" if filename_suffix is None else filename_suffix
    output_filepath = path.join(output_dir, "kd_latency_vary_" + key + "_" + suffix_to_use + ".csv")
    with open(output_filepath, "w") as output_file:
        output_file.write(key + ",min,max,median,average,standard_deviation\n")
        for value, micros_filepath in micros_files:
            latencies = [float(line) for line in open(micros_filepath)]
            latencies_without_warmup = latencies[num_warmup_trials:]
            output_file.write(",".join([
                str(value),
                str(min(latencies_without_warmup)),
                str(max(latencies_without_warmup)),
                str(numpy.median(latencies_without_warmup)),
                str(numpy.mean(latencies_without_warmup)),
                str(numpy.std(latencies_without_warmup))
            ]) + "\n")


def __main():
    """This program's entrypoint """
    run(parse_args(parser=None, validate_func=None), filename_suffix=None)


if __name__ == "__main__":
    __main()
