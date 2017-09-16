from __future__ import print_function
import csv
from collections import defaultdict
import argparse
import cPickle as pickle

def generate_shots_file(label_filename, shots_filename):
    frame_to_shot = defaultdict(int)
    shot_range_start = defaultdict(int)
    shot_range_end = defaultdict(int)
    shot_id = 0
    print("Generating shots file")
    print("Reading {} ...".format(label_filename))
    with open(label_filename) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = csv_reader.next()

        prev_frame_objs = defaultdict(int)
        cur_frame_objs = defaultdict(int)
        cur_frame_id = 0
        shot_range_start[0] = 0

        for row in csv_reader:
            frame_id = row[0]
            label = row[1]

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
        shots_dict = [frame_to_shot, shot_range_start, shot_range_end]
        print("Writing {} ...".format(shots_filename))
        pickle.dump(shots_dict, open(shots_filename, "wb"))

def dump_shots_file(shots_filename):
    [frame_to_shot, shot_range_start, shot_range_end] = pickle.load(open(shots_filename, "rb"))
    for shot_id, start in shot_range_start.iteritems():
        end = shot_range_end[shot_id]
        length = int(end) - int(start) + 1
        print("Shot ({}): length({}), range({}, {})".format(shot_id, length, start, end))

    for frame_id, shot_id in frame_to_shot.iteritems():
        print("Frame {} -> Shot {}".format(frame_id, shot_id))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Keyframe detector.");
    parser.add_argument("--verbose", dest="verbose", action="store_true",
            default=False, help="Verbose output")
    parser.add_argument("--dataset", dest="dataset_basename",
            help="Dataset base name (e.g., coral-reef-long)")
    parser.add_argument("--labels_to_shots", dest="labels_to_shots",
            action="store_true", default=False,
            help="Generate shots file from label file")
    parser.add_argument("--dump_shots", dest="dump_shots", action="store_true", 
            default=False, help="Dump shot to frame range")
    parser.add_argument("--keyframes", dest="keyframes_filename",
            help="File containing keyframes")
    parser.add_argument("--eval_keyframes", dest="eval_keyframes",
            action="store_true", default=False,
            help="Evaluate keyframes vs. ground truth")
    parser.add_argument("--start_frame", dest="start_frame", default=0,
                                 help="Start frame for evaluation")
    parser.add_argument("--end_frame", dest="end_frame", default=0,
                                 help="End frame for evaluation")
    args = parser.parse_args()

    if args.labels_to_shots:
        label_filename = args.dataset_basename + ".csv"
        shots_filename = args.dataset_basename + ".pkl"
        generate_shots_file(label_filename, shots_filename)

    if args.dump_shots:
        shots_filename = args.dataset_basename + ".pkl"
        dump_shots_file(shots_filename)

    if args.eval_keyframes:
        shots_filename = args.dataset_basename + ".pkl"
        keyframes_filename = args.keyframes_filename
        eval_keyframes(shots_filename, keyframes_filename)

if __name__ == "__main__":
    main()
