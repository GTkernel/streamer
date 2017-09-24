#!/usr/bin/env python3

import csv
from collections import defaultdict
import argparse
import pickle
import os
from pathlib import Path

def __parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Keyframe detector.");
    parser.add_argument("--verbose", dest="verbose",
            action="store_true", default=False,
            help="Print verbose")
    parser.add_argument("--outdir", dest="outdir", default=os.getcwd(),
            help="Destination directory for all output files")
    ## for converting dense labels to shots
    parser.add_argument("--labels", dest="labels_filepath",
            help="csv file with dense frame labels)")
    parser.add_argument("--extract_labels", dest="extract_labels",
            action="store_true", default=False,
            help="Extract labels for selected frame range")
    parser.add_argument("--labels_to_shots", dest="labels_to_shots",
            action="store_true", default=False,
            help="Generate shots file from label file")
    ## for keyframe evaluation
    parser.add_argument("--shots", dest="shots_filepath",
            help="Shots file (binary pickle file with shot ranges)")
    parser.add_argument("--print_shots", dest="print_shots",
            action="store_true", default=False,
            help="Print shot to frame range")
    parser.add_argument("--keyframes_dir", dest="keyframes_dir",
            help="Evaluate all keyframe (.log) files in the directory")
    parser.add_argument("--sampling", dest="sampling",
            action="store_true", default=False,
            help="Simulate uniform sampling for shot evaluation")
    parser.add_argument("--sampling_rates", dest="sampling_rates",
            default="0.5, 0.25, 0.125, 0.0625",
            help="Start frame for evaluation")
    parser.add_argument("--eval", dest="eval_keyframes",
            action="store_true", default=False,
            help="Evaluate keyframe shot coverage")
    parser.add_argument("--eval_outfile", dest="eval_outfile",
            default="dummy.csv", help="Evaluation output")
    parser.add_argument("--start_frame", dest="start_frame", default=0,
            help="Start frame for evaluation",
            type=int)
    parser.add_argument("--num_frames", dest="num_frames", default=0,
            help="Number of frames to evaluate",
            type=int)
    args = parser.parse_args()

    return args


def labels_to_shots(labels_filepath, shots_filepath, start_frame, num_frames):
    frame_to_shot = defaultdict(int)
    shot_range_start = defaultdict(int)
    shot_range_end = defaultdict(int)
    shot_id = 0
    print("Generating shots file from labels ...")
    print("Reading labels: {}".format(labels_filepath))
    with open(labels_filepath, "r") as labels_file:
        csv_reader = csv.reader(labels_file, delimiter=',')
        header = csv_reader.__next__()

        prev_frame_objs = defaultdict(int)
        cur_frame_objs = defaultdict(int)
        cur_frame_id = 0
        shot_range_start[0] = 0

        for row in csv_reader:
            frame_id = int(row[0])
            label = row[1]

            if num_frames != 0:
                # assumes that the csv is sorted by frame id
                if frame_id < start_frame:
                    continue
                if frame_id > start_frame + num_frames:
                    break

            if cur_frame_id == frame_id:
                cur_frame_objs[label] += 1
            else:
                frame_to_shot[cur_frame_id] = shot_id
                if cur_frame_id != 0 and cur_frame_objs != prev_frame_objs:
                    shot_range_end[shot_id] = cur_frame_id
                    shot_id += 1
                    shot_range_start[shot_id] = frame_id
               
                prev_frame_objs.clear()
                prev_frame_objs = cur_frame_objs.copy()
                cur_frame_objs.clear()
                cur_frame_id = frame_id
                cur_frame_objs[label] += 1

        shot_range_end[shot_id] = cur_frame_id

    with open(shots_filepath, "w") as shots_csv:
        print("Writing shots csv: {}".format(shots_filepath))
        csv_writer = csv.writer(shots_csv, delimiter=',')
        csv_writer.writerow(['shot_id', 'start_frame', 'end_frame'])
        for shot_id in range(0, len(shot_range_start)):
            csv_writer.writerow([shot_id, shot_range_start[shot_id], shot_range_end[shot_id]])

def read_shots(shots_filepath, start_frame, end_frame):
    frame_to_shot = defaultdict(int)
    shot_to_frame_range = defaultdict(range)

    print("Reading shots csv: {}".format(shots_filepath))
    with open(shots_filepath, "r") as shots_csv:
        csv_reader = csv.reader(shots_csv, delimiter=',')
        header = csv_reader.__next__() ## add checks
        for row in csv_reader:
            shot_id = int(row[0])
            shot_start = int(row[1])
            shot_end = int(row[2])
            if shot_start > end_frame:
                break
            shot_start = max(shot_start, start_frame)
            shot_end = min(shot_end, end_frame)
            shot_to_frame_range[shot_id] = (shot_start, shot_end)
            for frame_id in range(shot_start, shot_end + 1):
                frame_to_shot[frame_id] = shot_id
    return [shot_to_frame_range, frame_to_shot]

def __eval_keyframes(frame_to_shot, keyframes, start_frame, end_frame):
    # Number of shots in the given frame range
    shots = set()
    for frame_id in range(start_frame, end_frame):
        shots.add(frame_to_shot[frame_id])
    #print("Total shots: {}".format(len(shots)))

    # Shots covered by selected keyframes
    shot_coverage = set()
    for frame_id in keyframes:
        if frame_id >= start_frame and frame_id < end_frame:
            shot_coverage.add(frame_to_shot[frame_id])
    #print("KF shots: {}".format(len(shot_coverage)))
    #print("KF shots pct: {0:2f}".format(len(shot_coverage)/total_shots))
    return [shots, shot_coverage]

def __eval_output_row(csv_writer, kf_type, sel, buf_len, level, total_frames, kf_frames, total_shots, kf_shots):
    if total_shots:
        kf_shots_pct = "{0:.2f}".format(kf_shots/total_shots)
    else:
        kf_shots_pct = "{0:.2f}".format(0.00)
    row = [kf_type, sel, buf_len, level, total_frames, kf_frames, total_shots, kf_shots, kf_shots_pct]
    csv_writer.writerow(row)


def eval_sampling_keyframes(frame_to_shot, rate, start_frame, end_frame, csv_writer):
    win = int(1/rate)
    keyframes = [x for x in range(start_frame, end_frame) if x%win == 0]

    [shots, shot_coverage] = __eval_keyframes(frame_to_shot, keyframes, start_frame, end_frame)

    __eval_output_row(csv_writer, "sampling",
            rate, -1, -1,
            (end_frame - start_frame), len(keyframes),
            len(shots), len(shot_coverage))

def eval_kd_keyframes(frame_to_shot, kfile, start_frame, end_frame, csv_writer):
    keyframes = []
    with open(kfile, "r") as kf:
        for frame_num in kf:
            if frame_num.startswith("last frame processed:"):
                end_frame = int(frame_num.split(":")[1])
            else:
                keyframes.append(int(frame_num))

    [shots, shot_coverage] = __eval_keyframes(frame_to_shot, keyframes, start_frame, end_frame)

    kfile_basename = os.path.splitext(os.path.basename(kfile))[0]
    kfile_split = kfile_basename.split("_")
    __eval_output_row(csv_writer, "kd",
            kfile_split[3], kfile_split[4], kfile_split[2],
            (end_frame - start_frame), len(keyframes),
            len(shots), len(shot_coverage))

def handle_eval_keyframes(args, start_frame, end_frame):
    [shot_to_frame_range, frame_to_shot] = read_shots(args.shots_filepath, start_frame, end_frame)

    eval_outpath = os.path.join(args.outdir, args.eval_outfile)
    with open(eval_outpath, "w") as outfile:
        print("Writing keyframes eval: {}".format(eval_outpath))
        csv_writer = csv.writer(outfile, delimiter=',')
        header = ['kf_type', 'sel', 'buf_len', 'level', 'total_frames', 'kf_frames', 'total_shots', 'kf_shots', 'kf_shot_pct']
        csv_writer.writerow(header)

        ## Evaluate all keyframe files in the directory
        if args.keyframes_dir:
                keyframe_files = list(Path(args.keyframes_dir).glob("**/*.log"))
                for kfile in keyframe_files:
                    eval_kd_keyframes(frame_to_shot, str(kfile), start_frame, end_frame, csv_writer)

        ## Evaluate the performance of uniform sampling
        if args.sampling:
            sampling_rates = [float(x) for x in args.sampling_rates.split(",")]
            for rate in sampling_rates:
                eval_sampling_keyframes(frame_to_shot, rate, start_frame, end_frame, csv_writer)


def handle_extract_labels(args, start_frame, end_frame):
    dataset_basename = os.path.splitext(os.path.basename(args.labels_filepath))[0]
    extracted_labels_filepath = args.outdir + "/" + dataset_basename + "-s" + str(start_frame) + "-" + "e" + str(end_frame) + ".labels"

    with open(args.labels_filepath, "r") as rfile:
        with open(extracted_labels_filepath, "w") as wfile:
            csv_reader = csv.reader(rfile, delimiter=',')
            csv_writer = csv.writer(wfile, delimiter=',')
            header = csv_reader.__next__()
            csv_writer.writerow(header)
            for row in csv_reader:
                frame_id = int(row[0])
                if frame_id < start_frame:
                    continue
                if frame_id >= end_frame:
                    break
                csv_writer.writerow(row)


def handle_labels_to_shots(args, start_frame, end_frame):
    dataset_basename = os.path.splitext(os.path.basename(args.labels_filepath))[0]
    shots_filepath = args.outdir + "/" + dataset_basename + ".shots"
    labels_to_shots(args.labels_filepath, shots_filepath, start_frame, end_frame)

def handle_print_shots(args, start_frame, end_frame, verbose=False):
    [shot_to_frame_range, frame_to_shot] = read_shots(args.shots_filepath, start_frame, end_frame)

    for shot_id, (shot_start, shot_end) in shot_to_frame_range.items():
        length = int(shot_end) - int(shot_start) + 1
        print("Shot ({}): length({}), range({}, {})".format(shot_id, length, shot_start, shot_end))

    total_shots = len(shot_to_frame_range)
    print("#shots (shot_to_frame_range): {}".format(len(shot_to_frame_range)))
    shots = set()
    for frame_id in range(start_frame, end_frame):
        shot_id = frame_to_shot[frame_id]
        shots.add(shot_id)
        print("Frame {} -> Shot {}".format(frame_id, shot_id))
    total_shots = len(shots)
    print("#shots (frame_to_shot): {}".format(total_shots))

def main():
    args = __parse_args()

    start_frame = int(args.start_frame)
    num_frames = int(args.num_frames)
    end_frame = start_frame + num_frames

    if args.extract_labels:
        handle_extract_labels(args, start_frame, end_frame)

    if args.labels_to_shots:
        handle_labels_to_shots(args, start_frame, end_frame)

    if args.print_shots:
        handle_print_shots(args, start_frame, end_frame)

    if args.eval_keyframes:
        handle_eval_keyframes(args, start_frame, end_frame)

if __name__ == "__main__":
    main()
