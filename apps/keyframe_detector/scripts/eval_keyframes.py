from __future__ import print_function
import csv
from collections import defaultdict
import argparse
import cPickle as pickle
import os

def __parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Keyframe detector.");
    parser.add_argument("--verbose", dest="verbose",
            action="store_true", default=False,
            help="Print verbose")
    parser.add_argument("--workdir", dest="workdir", default=os.getcwd(),
            help="Working directory (stores all files and results)",
            required=True)
    parser.add_argument("--labels", dest="labels_filepath",
            help="Labels file (csv file with dense frame-wise labels)")
    parser.add_argument("--labels_to_shots", dest="labels_to_shots",
            action="store_true", default=False,
            help="Generate shots file from label file")
    parser.add_argument("--shots", dest="shots_filepath",
            help="Shots file (binary pickle file with shot ranges)")
    parser.add_argument("--print_shots", dest="print_shots", action="store_true",
            default=False, help="Print shot to frame range")
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

def generate_shots_file(label_filepath, shots_filepath, start_frame=0, num_frames=0):
    frame_to_shot = defaultdict(int)
    shot_range_start = defaultdict(int)
    shot_range_end = defaultdict(int)
    shot_id = 0
    print("Generating shots file ...")
    print("Reading labels: {}".format(label_filepath))
    with open(label_filepath) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = csv_reader.next()

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
        shots_dict = [frame_to_shot, shot_range_start, shot_range_end]
        print("Writing shots: {}".format(shots_filepath))
        pickle.dump(shots_dict, open(shots_filepath, "wb"))

def print_shots(shots_filepath, start_frame=0, num_frames=0, verbose=False):
    [frame_to_shot, shot_range_start, shot_range_end] = pickle.load(open(shots_filepath, "rb"))
    if verbose == True:
        for shot_id, start in shot_range_start.iteritems():
            end = shot_range_end[shot_id]
            length = int(end) - int(start) + 1
            print("Shot ({}): length({}), range({}, {})".format(shot_id, length, start, end))
        #for frame_id, shot_id in frame_to_shot.iteritems():
        #    print("Frame {} -> Shot {}".format(frame_id, shot_id))

    total_shots = len(shot_range_start)
    if num_frames != 0:
        shots = set()
        for frame_id in range(start_frame, start_frame + num_frames):
            shot_id = frame_to_shot[frame_id]
            shots.add(shot_id)
            print("Frame {} -> Shot {}".format(frame_id, shot_id))
        total_shot = len(shots)
    print("Total shots: {}".format(len(shot_range_start)))

def eval_keyframes(shots_filepath, keyframes_filepath, shotcov_filepath, start_frame=0, num_frames=0):
    [frame_to_shot, shot_range_start, shot_range_end] = pickle.load(open(shots_filepath, "rb"))
    if num_frames == 0:
        num_frames = len(shot_range_start)
    end_frame = start_frame + num_frames
    # Number of shots in the given frame range
    shots = set()
    for frame_id in range(start_frame, end_frame):
        shots.add(frame_to_shot[frame_id])
    # Shots covered by selected keyframes
    shot_coverage = set()
    with open(keyframes_filepath) as kf:
        for frame_num in kf:
            frame_id = int(frame_num)
            if frame_id >= start_frame and frame_id < end_frame:
                shot_id = frame_to_shot[frame_id]
                shot_coverage.add(shot_id)
    print("Total shots: {}".format(len(shots)))
    print("Shots covered: {}".format(len(shot_coverage)))
    print("Write shotcov: {}".format(shotcov_filepath))
    pickle.dump([shots, shot_coverage], open(shotcov_filepath, "wb"))

def main():
    args = __parse_args()
    workdir = args.workdir

    start_frame = int(args.start_frame)
    num_frames = int(args.num_frames)

    if args.labels_to_shots:
        dataset_basename = os.path.splitext(os.path.basename(args.labels_filepath))[0]
        shots_filepath = workdir + "/" + dataset_basename + ".shots"
        generate_shots_file(args.labels_filepath, shots_filepath, start_frame, num_frames)

    if args.print_shots:
        print_shots(args.shots_filepath, start_frame, num_frames)

    if args.eval_keyframes:
        keyframes_basename = os.path.splitext(os.path.basename(args.keyframes_filepath))[0]
        shotcov_filepath = workdir + "/" + keyframes_basename + ".shotcov"
        eval_keyframes(args.shots_filepath, args.keyframes_filepath, shotcov_filepath, start_frame, num_frames)

if __name__ == "__main__":
    main()
