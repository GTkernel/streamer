#!/bin/bash
# Args: $1 - directory containing directories of keyframe results to evaluate
#       $2 - path to train_timestamps.txt file
#       $3 - path to output directory

mkdir -p $3
for dir in $(ls $1); do
  echo $dir
  python plot_1d_keyframe.py "$1/$dir/keyframe_buffer*" $2 $3/$dir.png
done
