#!/usr/bin/python3

import argparse
import os
from os import path


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir-1", help="The first results directory.", required=True,
                        type=str)
    parser.add_argument("--results-dir-2", help="The second results directory.", required=True,
                        type=str)
    parser.add_argument("--output-dir", help="The output directory.", required=True, type=str)
    parser.add_argument("--first-is-veryfast-slow",
                        action="store_true",
                        default=False,
                        help=("Whether the first results dir corresponds to the veryfast_slow "
                              "video."))
    parser.add_argument("--leeway-frames",
                        default=0,
                        help=("The number of frames of leeway to use when deciding whether two "
                              "keyframes are the same."),
                        required=False,
                        type=int)


    return parser.parse_args()


def __parse_dir(dir_path):
    results = {}
    # Loop over every experiment dir...
    for subdir_name in os.listdir(dir_path):
        subdir_path = path.join(dir_path, subdir_name)
        if path.isdir(subdir_path):
            tokens = subdir_name.split("_")
            assert len(tokens) == 3, "Subdir name must be of the form \"sel_buf-len_levels\"!"
            sel = float(tokens[0])
            buf_len = int(tokens[1])
            levels = int(tokens[2])

            levels_results = []
            # Loop over every results file...
            for i in range(levels):
                keyframes = []
                last_frame_processed = 0

                results_filepath = path.join(
                    subdir_path, "keyframe_buffer_{}_{}_{}.log".format(i, sel, buf_len))

                # Loop over every line in the results file...
                for line in  open(results_filepath):
                    if line.startswith("last frame processed:"):
                        last_frame_processed = int(line.split(":")[1].split(" ")[1])
                    else:
                        keyframes.append(int(line))

                levels_results.append((keyframes, last_frame_processed))
            results[(sel, buf_len)] = levels_results
    return results


def __frame_id_svf_to_vfs(frame_id, svf_split_id, vfs_split_id):

    if frame_id < svf_split_id:
        result = frame_id / 4.0
    else:
        result = vfs_split_id + (frame_id - svf_split_id) * 4

    # print("svf->vfs: {} -> {}".format(frame_id, result))
    return result


def __frame_id_vfs_to_svf(frame_id, svf_split_id, vfs_split_id):
    if frame_id < vfs_split_id:
        result = frame_id * 4
    else:
        result = svf_split_id + (frame_id - vfs_split_id) / 4.0

    # print("vfs->svf: {} -> {}".format(frame_id, result))
    return result


def __find_closest(frames, target_frame):
    closest = frames[0]
    dist = abs(target_frame - closest)

    for frame in frames:
        frame_dist = abs(target_frame - frame)
        if frame_dist < dist:
            closest = frame
            dist = frame_dist

    return closest


def __compare_keyframes(keyframes_1, keyframes_2, first_is_veryfast_slow, leeway):
    if len(keyframes_1) == 0:
        return 0

    ground_truth = keyframes_1

    if first_is_veryfast_slow:
        # convert second to veryfast_slow
        keyframes_converted = [__frame_id_svf_to_vfs(keyframe, svf_split_id=560, vfs_split_id=140)
                               for keyframe in keyframes_2]
    else:
        # convert second to slow_veryfast
        keyframes_converted = [__frame_id_vfs_to_svf(keyframe, svf_split_id=560, vfs_split_id=140)
                               for keyframe in keyframes_2]

    #print("pairs:")
    #for k1, k2 in zip(keyframes_converted, ground_truth):
    #    print("{} : {}".format(k1, k2))

    # find frame id of closest frame in ground truth
    # [ (key, closest), ... ]
    closest_pairs = [(keyframe, __find_closest(ground_truth, keyframe))
                     for keyframe in keyframes_converted]
    #print("closest_pairs:")
    #for k, closest in closest_pairs:
    #    print("{} : {}".format(k, closest))

    # The error (after subtracting leeway) between each keyframe and its corresponding frame in the
    # ground truth.
    errors = [abs(keyframe - closest) - leeway for keyframe, closest in closest_pairs]

    total_error = sum(errors)
    #normalized_error = total_error / float(len(ground_truth))
    #print("total_error: {}".format(total_error))
    #print("normalized_error: {}".format(normalized_error))
    return total_error  # normalized_error


def __compare(data_1, data_2, first_is_veryfast_slow, leeway_frames):
    # (sel, buf_len) -> [level 0 similarity, level 1 similarity, ...]
    # ...
    results = {}
    levels = 0
    for key, data_1_levels_results in data_1.items():
        data_2_levels_results = data_2[key]
        similarity_levels_results = []

        if levels == 0:
            levels = len(data_1_levels_results)
        else:
            assert len(data_1_levels_results) == levels
            assert len(data_2_levels_results) == levels

        for level_results_1, level_results_2 in zip(data_1_levels_results, data_2_levels_results):
            keyframes_1, _ = level_results_1
            keyframes_2, _ = level_results_2

            assert len(keyframes_1) == len(keyframes_2)

            similarity_levels_results.append(
                __compare_keyframes(keyframes_1, keyframes_2, first_is_veryfast_slow,
                                    leeway_frames))

        results[key] = similarity_levels_results

    results_2 = {}
    # For each level...
    for level in range(levels):
        # { sel -> { buf_len -> similarity } }
        results_for_level = {}
        # Add that level's result for each (sel, buf_len) pair...
        for (sel, buf_len), similarities in results.items():
            if sel not in results_for_level:
                results_for_level[sel] = {}
            results_for_level[sel][buf_len] = similarities[level]
        results_2[level] = results_for_level

    results_3 = {}
    for level, results_for_level in results_2.items():
        results_3[level] = []
        on_first = True

        for sel, results_for_sel in sorted(results_for_level.items(), key=lambda pair: pair[0]):
            results_list = [sel]
            title_line = ["sel"]

            for buf_len, result in sorted(results_for_sel.items(), key=lambda pair: pair[0]):
                results_list.append(result)
                title_line.append(buf_len)
            results_3[level].append(results_list)

            if on_first:
                results_3[level].insert(0, title_line)
                on_first = False

    return results_3


def __main():
    args = __parse_args()
    for level, results_per_level in __compare(
            __parse_dir(args.results_dir_1),
            __parse_dir(args.results_dir_2),
            args.first_is_veryfast_slow, args.leeway_frames).items():
        with open(path.join(args.output_dir,
                            "rotational_results_{}.csv".format(level)), "w") as results_file:
            for line in results_per_level:
                results_file.write("{}\n".format(",".join([str(x) for x in line])))


if __name__ == "__main__":
    __main()
