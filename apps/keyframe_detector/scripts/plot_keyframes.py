from tsne_utils import plot_tsne_with_keyframe_train_windows
from tsne_utils import plot_tsne_with_keyframes
from tsne_utils import plot_tsne_with_mouseover_viz_and_keyframes
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_keyframes(keyframe_path):
  with open(keyframe_path, 'r') as f:
    elems = [line for line in f]
    keyframes = [int(i) for i in elems[:-1]]
    last_frame = int(elems[-1].split()[-1])
  return keyframes

def load_train_frames(train_frame_path):
  with open(train_frame_path, 'r') as f:
    return [(int(line.split()[0]), int(line.split()[1])) for line in f]

if __name__ == '__main__':
  data = np.load(sys.argv[1])
  embedded_data = np.load(sys.argv[2])
  keyframes = load_keyframes(sys.argv[3])
  train_frames = load_train_frames(sys.argv[4])
  filepath = None
  if len(sys.argv) >= 6:
    filepath = sys.argv[5]

  plot_tsne_with_keyframe_train_windows(embedded_data, data, keyframes, train_frames, filepath=filepath)
  # plot_tsne_with_keyframes(embedded_data, data, keyframes, train_frames)
  # plot_tsne_with_mouseover_viz_and_keyframes(embedded_data, data, keyframes)
