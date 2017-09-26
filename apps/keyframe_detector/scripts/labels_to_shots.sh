#!/bin/bash

dataset=$1
shots_type=$2
results_dir="/home/dulloor/filterforward/results"
labels_file=""
shots_file=""
start_frame=0
num_frames=0

if [ "$shots_type" == "sparse" ]; then
	results_dir="$results_dir/sparse"
else
	results_dir="$results_dir/dense"
fi

if [ "$dataset" == "coral" ]; then
	labels_file="$results_dir/coral-reef-long-s0-d5010/coral-reef-long-s0-e5000.labels"
	results_dir="$results_dir/coral-reef-long-s0-d5010"
	start_frame=0
	num_frames=5000
elif [ "$dataset" == "jackson" ]; then
	labels_file="$results_dir/jackson-town-square-s0-d5010/jackson-town-square-s0-e5000.labels"
	results_dir="$results_dir/jackson-town-square-s0-d5010"
	start_frame=0
	num_frames=5000
else
	echo "Usage: ./labels_to_shtos.sh (coral | jackson) (dense | sparse)" 
	exit
fi

echo "dataset: $dataset"
echo "shots type: $shots_type"
echo "labels_file: $labels_file"
echo "shots_file: $shots_file"
echo "start_frame: $start_frame"
echo "num_frames: $num_frames"

eval_cmd="./eval_keyframes.py --outdir ${results_dir} --start_frame ${start_frame} --num_frames ${num_frames} --labels ${labels_file} --labels_to_shots"

if [ "$shots_type" == "sparse" ]; then
	eval_cmd="$eval_cmd --sparse_shots"
fi

echo "Executing: $eval_cmd"
$eval_cmd
