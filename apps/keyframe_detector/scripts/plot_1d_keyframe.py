from matplotlib.pyplot import cm
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_keyframes(keyframe_path):
  with open(keyframe_path, 'r') as f:
    elems = [line for line in f]
    keyframes = [int(i) for i in elems[:-1]]
    last_frame = int(elems[-1].split()[-1])
  return keyframes, last_frame

def load_train_frames(train_frame_path):
  with open(train_frame_path, 'r') as f:
    return [(int(line.split()[0]), int(line.split()[1])) for line in f]

def plot_keyframes_and_train_intervals(
    keyframes_title, keyframe_names, keyframes, train_frames, last_frames, savepath):
  max_x = max(list(itertools.chain(*keyframes)) + [x[1] for x in train_frames])
  # Eliminate any levels that don't have any contents.
  if len(filter(lambda x: len(x) > 0, keyframes)) < len(keyframes):
    new_keyframes, new_keyframe_names, new_last_frames = [], [], []
    for i, keyframes in enumerate(keyframes):
      if len(keyframes) > 0:
        new_keyframes.append(keyframes)
        new_keyframe_names.append(keyframe_names[i])
        new_last_frames.append(last_frames[i])
    keyframes, keyframe_names, last_frames = new_keyframes, new_keyframe_names, new_last_frames

  # Find the topmost level that catches all the trains.
  for i in range(len(keyframes)):
    uniform_idx = -1 - i
    contains_trains = [len(filter(lambda y: y in range(tframe_0, tframe_1), keyframes[uniform_idx])) > 0 for tframe_0, tframe_1 in train_frames]
    if contains_trains.count(True) == len(contains_trains):    
      break

  # Plot uniform baseline as well.
  every_nth_frame = np.arange(0, len(keyframes[uniform_idx])) * int(max_x / len(keyframes[uniform_idx]))
  keyframe_names.append('Uniform baseline with no. points == %s' % keyframe_names[uniform_idx])
  keyframes.append(every_nth_frame)

  colors = cm.rainbow(np.linspace(0, 1, len(keyframes)))
  for i, (name, keyframe) in enumerate(zip(keyframe_names, keyframes)):
    plt.scatter(keyframe, [i] * len(keyframe), alpha=0.3, c=colors[i])
    plt.text(0.05 * max_x, i + 0.1, name)

  colors = cm.rainbow(np.linspace(0, 1, len(train_frames)))
  for i, (train_start, train_end) in enumerate(train_frames):
    plt.axvline(x=train_start, c=colors[i])
    plt.axvline(x=train_end, c=colors[i])

  plt.scatter(last_frames, [i for i in range(len(last_frames))], marker='x', c='r')

  plt.xlim([0, max_x])
  plt.xlabel('Frame number')
  plt.ylabel('Buffer levels')
  plt.title('Keyframes at varying buffer levels (%s)' % keyframes_title) 
  # plt.show()
  plt.savefig(savepath)

if __name__ == '__main__':
  keyframe_files = glob.glob(sys.argv[1])
  keyframes, last_frames = zip(*[load_keyframes(path) for path in keyframe_files])
  keyframes, last_frames = list(keyframes), list(last_frames)
  train_frames = load_train_frames(sys.argv[2])
  savepath = sys.argv[3]

  keyframes_title = sys.argv[1].split('/')[-2] 
  keyframe_names = [x.split('/')[-1] for x in keyframe_files]
  plot_keyframes_and_train_intervals(
      keyframes_title, keyframe_names, keyframes, train_frames, last_frames, savepath)
