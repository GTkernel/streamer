#!/usr/bin/env python3

import csv
from collections import defaultdict
import argparse
import pickle
import os

def __parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Keyframe detector.");
    parser.add_argument("--verbose", dest="verbose",
            action="store_true", default=False,
            help="Print verbose")
    parser.add_argument("--workdir", dest="workdir", default=os.getcwd(),
            help="Working directory (stores all files and results)",
            required=True)
    parser.add_argument("--savepkl", dest="save_pkl", action="store_true",
            default=False,
            help="Save shot files and results as pickle files (default is csv)")
    parser.add_argument("--labels", dest="labels_filepath",
            help="Labels file (csv file with dense frame-wise labels)")
    parser.add_argument("--extract_labels", dest="extract_labels",
            action="store_true", default=False,
            help="Extract labels for selected frame range")
    parser.add_argument("--labels_to_shots", dest="labels_to_shots",
            action="store_true", default=False,
            help="Generate shots file from label file")
    parser.add_argument("--shots", dest="shots_filepath",
            help="Shots file (binary pickle file with shot ranges)")
    parser.add_argument("--print_shots", dest="print_shots",
            action="store_true", default=False,
            help="Print shot to frame range")
    parser.add_argument("--keyframes", dest="keyframes_filepath",
            help="Keyframes file")
    parser.add_argument("--eval", dest="eval_keyframes",
            action="store_true", default=False,
            help="Evaluate keyframe shot coverage")
    parser.add_argument("--start_frame", dest="start_frame", default=0,
                                 help="Start frame for evaluation")
    parser.add_argument("--num_frames", dest="num_frames", default=0,
                                 help="Number of frames to evaluate")
    args = parser.parse_args()

    return args


def extract_labels(labels_filepath, extracted_labels_filepath, start_frame, end_frame):
    with open(labels_filepath, "r") as rfile:
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


def labels_to_shots(labels_filepath, shots_filepath, start_frame=0, num_frames=0, save_pkl=False):
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

    shots_csvpath = shots_filepath + ".csv"
    with open(shots_csvpath, "w") as shots_csv:
        print("Writing shots csv: {}".format(shots_csvpath))
        csv_writer = csv.writer(shots_csv, delimiter=',')
        csv_writer.writerow(['shot_id', 'start_frame', 'end_frame'])
        for shot_id in range(0, len(shot_range_start)):
            csv_writer.writerow([shot_id, shot_range_start[shot_id], shot_range_end[shot_id]])

    if save_pkl == True:
        shot_range = defaultdict(range)
        for shot_id in range(0, len(shot_range_start)):
            shot_range[shot_id] = (shot_range_start[shot_id], shot_range_end[shot_id])
        shots_pklpath = shots_filepath + ".pkl"
        print("Writing shots pkl: {}".format(shots_pklpath))
        shots_dict = [shot_to_frame_range, frame_to_shot]
        pickle.dump(shots_dict, open(shots_pklpath, "wb"))

def read_shots_csv(shots_csvpath, start_frame, end_frame):
    frame_to_shot = defaultdict(int)
    shot_to_frame_range = defaultdict(range)

    print("Reading shots csv: {}".format(shots_csvpath))
    with open(shots_csvpath, "r") as shots_csv:
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

def read_shots_file(shots_filepath, start_frame, end_frame):
    file_ext = os.path.splitext(shots_filepath)[1]
    if file_ext == '.csv':
        [shot_to_frame_range, frame_to_shot] = read_shots_csv(shots_filepath, start_frame, end_frame)
    elif file_ext == '.pkl':
        [shot_to_frame_range, frame_to_shot] = pickle.load(open(shots_filepath, "rb"))
        for shot_id, (shot_start, shot_end) in shot_range.iteriterms():
            if shot_end < start_frame or shot_start > end_frame:
                del shot_to_frame_range[shot_id]
    return [shot_to_frame_range, frame_to_shot]

def print_shots(shots_filepath, start_frame, num_frames, verbose=False):
    end_frame = start_frame + num_frames

    [shot_to_frame_range, frame_to_shot] = read_shots_file(shots_filepath, start_frame, end_frame)

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

def eval_keyframes(shots_filepath, keyframes_filepath, shotcov_filepath, start_frame, num_frames, save_pkl=False):
    end_frame = start_frame + num_frames

    [shot_to_frame_range, frame_to_shot] = read_shots_file(shots_filepath, start_frame, end_frame)
    # Number of shots in the given frame range
    shots = set()
    for frame_id in range(start_frame, end_frame):
        shots.add(frame_to_shot[frame_id])
    # Shots covered by selected keyframes
    shot_coverage = set()
    with open(keyframes_filepath, "r") as kf:
        for frame_num in kf:
            frame_id = int(frame_num)
            if frame_id >= start_frame and frame_id < end_frame:
                shot_id = frame_to_shot[frame_id]
                shot_coverage.add(shot_id)
    print("Total shots: {}".format(len(shots)))
    print("Shots covered: {}".format(len(shot_coverage)))
    if save_pkl == True:
        print("Write shotcov: {}".format(shotcov_filepath))
        pickle.dump([shots, shot_coverage], open(shotcov_filepath, "wb"))

def main():
    args = __parse_args()
    workdir = args.workdir

    start_frame = int(args.start_frame)
    num_frames = int(args.num_frames)

    if args.extract_labels:
        end_frame = start_frame + num_frames
        dataset_basename = os.path.splitext(os.path.basename(args.labels_filepath))[0]
        extracted_labels_filepath = workdir + "/" + dataset_basename + "-s" + str(start_frame) + "-" + "e" + str(start_frame + num_frames) + ".labels.csv"
        extract_labels(args.labels_filepath, extracted_labels_filepath, start_frame, end_frame)

    if args.labels_to_shots:
        dataset_basename = os.path.splitext(os.path.basename(args.labels_filepath))[0]
        shots_filepath = workdir + "/" + dataset_basename + ".shots"
        labels_to_shots(args.labels_filepath, shots_filepath, start_frame, num_frames)

    if args.print_shots:
        print_shots(args.shots_filepath, start_frame, num_frames)

    if args.eval_keyframes:
        keyframes_basename = os.path.splitext(os.path.basename(args.keyframes_filepath))[0]
        shotcov_filepath = workdir + "/" + keyframes_basename + ".shotcov"
        eval_keyframes(args.shots_filepath, args.keyframes_filepath, shotcov_filepath, start_frame, num_frames)

if __name__ == "__main__":
    main()
