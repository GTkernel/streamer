#!/bin/bash

dataset=$1
results_dir="/home/dulloor/filterforward/results"
shots_file=""
keyframes_dir=""
start_frame=0
num_frames=0
results_file=""

if [ "$dataset" == "coral" ]; then
	shots_file="$results_dir/coral-reef-long-s0-d5010/coral-reef-long-s0-e5000.shots"
	keyframes_dir="/disk3/noscope_data/coral_reef_long_s0_d5010/kd/3"
	start_frame=0
	num_frames=5000
	results_file="coral-reef-long-s0-e5000.eval"
elif [ "$dataset" == "jackson" ]; then
	shots_file="$results_dir/jackson-town-square-s0-d5010/jackson-town-square-s0-e5000.shots"
	keyframes_dir="/disk3/noscope_data/jackson_town_square_s0_d5010/kd/3"
	start_frame=0
	num_frames=5000
	results_file="jackson-town-square-s0-e5000.eval"
elif [ "$dataset" == "train1" ]; then
	shots_file="/disk3/train/train_cam_medium/train_cam_medium_shots.txt"
	keyframes_dir="/disk3/train/train_cam_medium/kd/6"
	results_file="train-cam-medium-1.eval"
	start_frame=0
	num_frames=10838
elif [ "$dataset" == "train2" ]; then
	shots_file="/disk3/train/train_cam_medium_2/train_cam_medium_2_shots.txt"
	keyframes_dir="/disk3/train/train_cam_medium_2/kd/2"
	results_file="train-cam-medium-2.eval"
	start_frame=0
	num_frames=12600
elif [ "$dataset" == "rotational" ]; then
	shots_file="/disk3/rotational/rotational.shots"
	keyframes_dir="/disk3/rotational/kd/1"
	results_file="rotational.eval"
else
	echo "Usage: ./evaluate.sh (coral | jackson | train1 | train2 | rotational)" 
	exit
fi

echo "dataset: $dataset"
echo "results_dir: $results_dir"
echo "shots_file: $shots_file"
echo "keyframes_dir: $keyframes_dir"
echo "start_frame: $start_frame"
echo "num_frames: $num_frames"
echo "results_file: $results_file"

eval_cmd="./eval_keyframes.py --outdir ${results_dir} --start_frame ${start_frame} --num_frames ${num_frames} --shots ${shots_file} --keyframes_dir ${keyframes_dir} --eval_outfile ${results_file} --eval"
echo "Executing: $eval_cmd"
$eval_cmd
#./eval_keyframes.py --outdir "${results_dir}" --start_frame "${start_frame}" --num_frames "${num_frames}" --shots "${shots_file}" --keyframes_dir "${keyframe_dir}" --eval_outfile "${results_file}" --eval

