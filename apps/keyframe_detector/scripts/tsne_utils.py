# Functions to apply and visualize t-SNE results.
from matplotlib.pyplot import cm
from matplotlib.widgets import Button
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn.manifold import TSNE 
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

tsne_filenames = {
  'data': 'data.npy', 'labels': 'labels.npy', 'embedded_data': 'embedded_data.npy'
}

class tsnePoint(object):
  def __init__(self, coordinates, timestamp, cluster_no=None):
    self.coordinates = coordinates
    self.timestamp = timestamp
    self.cluster_no = cluster_no

def load_mnist():
  from sklearn.datasets import fetch_mldata
  mnist = fetch_mldata('MNIST original', data_home='data')
  return mnist.data, mnist.target

def apply_tsne(data):
  return TSNE(n_components=2, verbose=1).fit_transform(data)

def plot_tsne(data, labels):
  unique_labels = np.unique(labels)
  colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
  for i, label in enumerate(np.unique(labels)):
    class_data = data[np.where(labels == label)]
    plt.scatter(class_data[:, 0], class_data[:, 1], c=colors[i], label=str(label))
  plt.legend()
  plt.title('t-SNE embedding by class')
  plt.show()

def plot_tsne_with_mouseover_viz(embedded_data, labels, data):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  # Plot data in first subplot.
  unique_labels = np.unique(labels)
  colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
  for i, label in enumerate(np.unique(labels)):
    class_data = embedded_data[np.where(labels == label)]
    ax1.plot(class_data[:, 0], class_data[:, 1], c=colors[i], label=str(label), picker=5, marker='o', linestyle='None')
  leg = ax1.legend()
  leg.get_frame().set_alpha(0.5)
  ax1.set_title('t-SNE embedding by class')

  # Set up second subplot for viz.
  ax2.imshow(np.zeros((28, 28, 3)))
  embed_to_data = {(x[0], x[1]): data[i] for i, x in enumerate(embedded_data)}
  def redraw_data(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    point = (xdata[ind][0], ydata[ind][0])
    hovered_data = embed_to_data[point]
    ax2.imshow(hovered_data)
    fig.canvas.draw()
  
  fig.canvas.mpl_connect('pick_event', redraw_data)
  plt.show()

def plot_tsne_with_mouseover_viz_no_label(embedded_data, data):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  # Plot data in second subplot.
  cax = ax2.scatter(embedded_data[:, 0], embedded_data[:, 1], c=[float(i) for i in range(len(data))], picker=True, alpha=0.3)
  cbar = fig.colorbar(cax)
  ax2.set_title('t-SNE embedding by frame number')

  # Set up first subplot for viz.
  ax1.imshow(np.zeros((28, 28, 3)))
  embed_to_data = {(x[0], x[1]): data[i] for i, x in enumerate(embedded_data)}
  def redraw_data(event):
    ind = event.ind
    xy = event.artist.get_offsets()
    nearby_points = xy[ind]
    mousepoint = np.array([event.mouseevent.xdata, event.mouseevent.ydata])
    nearest_point_idx = np.argmin(np.linalg.norm(nearby_points - mousepoint))
    nearest_point = nearby_points[nearest_point_idx]
    hovered_data = embed_to_data[(nearest_point[0], nearest_point[1])]
    ax1.set_title('Frame No. %d' % ind[nearest_point_idx])
    ax1.imshow(hovered_data)
    fig.canvas.draw()

  fig.canvas.mpl_connect('pick_event', redraw_data)
  plt.show()

def plot_tsne_with_mouseover_viz_select_clusters(embedded_data, data):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  # Plot data in second subplot.
  cax = ax2.scatter(embedded_data[:, 0], embedded_data[:, 1], c=[float(i) for i in range(len(data))], picker=True, alpha=0.3)
  cbar = fig.colorbar(cax)
  ax2.set_title('t-SNE embedding by frame number')

  # Set up first subplot for viz.
  ax1.imshow(np.zeros((28, 28, 3)))
  embed_to_data = {(x[0], x[1]): data[i] for i, x in enumerate(embedded_data)}

  # Set up state variables.
  class SelectClustersObj(object):
    selecting_clusters = False
    selected_points = []
    selected_point_objs = []
    lines = None
    clusters = []

    def activate_selection(self, event):
      self.selecting_clusters = not self.selecting_clusters
      if self.selecting_clusters:
        ax2.set_title('t-SNE embedding by frame number (selection mode on)')
      else:
        ax2.set_title('t-SNE embedding by frame number')
      fig.canvas.draw()

    def find_bounded_points(self):
      polygon = Polygon([[x[0], x[1]] for x in self.selected_points])
      bounded_points = []
      for i, point in enumerate(embedded_data):
        if polygon.contains(Point(point[0], point[1])):
          bounded_points.append(tsnePoint(point, i))
      return bounded_points

    def redraw_lines(self):
      assert len(self.selected_points) > 0
      if self.lines is not None: 
        self.lines.remove()
      xdata = [x[0] for x in self.selected_points]
      ydata = [x[1] for x in self.selected_points]
      xdata.append(self.selected_points[0][0])
      ydata.append(self.selected_points[0][1])
      # plot returns an array, so get the first one.
      self.lines = ax2.plot(xdata, ydata, c='k')[0]
      fig.canvas.draw()

    def select_cluster(self, event):
      if not self.selecting_clusters:
        return
      elif len(self.selected_points) < 3:
        print('Need at least 3 points to select cluster.')
        return
      # Find points that fall within the polygon.
      bounded_points = self.find_bounded_points()
      for i in range(len(bounded_points)):
        bounded_points[i].cluster_no = len(self.clusters)
      self.clusters.append(bounded_points)

      xs = [x.coordinates[0] for x in bounded_points]
      ys = [x.coordinates[1] for x in bounded_points]
      print("Added %d bounded_points." % len(bounded_points))
      print("x: [%f, %f], y: [%f, %f]" % (min(xs), max(xs), min(ys), max(ys)))

      self.activate_selection(event) # Deactivate selection mode.
      self.reset_select(event, hard_reset=False)

    def reset_select(self, event, hard_reset=True):
      if hard_reset:
        if self.lines is not None:
          self.lines.remove()
        for point_obj in self.selected_point_objs:
          point_obj.remove()
      self.lines = None
      self.selected_points = []
      self.selected_point_objs = []
      fig.canvas.draw()

  select_clusters_callback = SelectClustersObj()
  selection_button = Button(
    plt.axes([0.1, 0.05, 0.1, 0.075]), 'Select Mode')
  selection_button.on_clicked(select_clusters_callback.activate_selection)
  cluster_select_button = Button(
    plt.axes([0.25, 0.05, 0.1, 0.075]), 'Select Cluster')
  cluster_select_button.on_clicked(select_clusters_callback.select_cluster)
  reset_select_button = Button(
    plt.axes([0.35, 0.05, 0.1, 0.075]), 'Reset')
  reset_select_button.on_clicked(select_clusters_callback.reset_select)

  def mark_cluster_point(event):
    if not select_clusters_callback.selecting_clusters:
      return
    elif event.inaxes not in [ax2]: # don't include click outside t-SNE plot
      return
    mousepoint = np.array([event.xdata, event.ydata])
    # Returns list of lines, so grab the first one.
    select_point = ax2.plot(event.xdata, event.ydata, marker='s', c='k')[0]
    # Update state variable.
    #  1. Add points and lines.
    #  2. Draw new lines.
    select_clusters_callback.selected_points.append(mousepoint)
    select_clusters_callback.selected_point_objs.append(select_point)
    select_clusters_callback.redraw_lines()
    fig.canvas.draw()

  def redraw_data(event):
    if select_clusters_callback.selecting_clusters:
      return
    ind = event.ind
    xy = event.artist.get_offsets()
    nearby_points = xy[ind]
    mousepoint = np.array([event.mouseevent.xdata, event.mouseevent.ydata])
    nearest_point_idx = np.argmin(np.linalg.norm(nearby_points - mousepoint))
    nearest_point = nearby_points[nearest_point_idx]
    hovered_data = embed_to_data[(nearest_point[0], nearest_point[1])]
    ax1.set_title('Frame No. %d' % ind[nearest_point_idx])
    ax1.imshow(hovered_data)
    fig.canvas.draw()

  fig.canvas.mpl_connect('button_press_event', mark_cluster_point)
  fig.canvas.mpl_connect('pick_event', redraw_data)
  plt.show()

  return select_clusters_callback.clusters

def plot_tsne_with_mouseover_viz_and_keyframes(embedded_data, data, keyframes):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  # Plot data in second subplot.
  cax = ax2.scatter(embedded_data[:, 0], embedded_data[:, 1], c=[float(i) for i in range(len(data))], picker=True, alpha=0.3)
  cbar = fig.colorbar(cax)
  ax2.set_title('t-SNE embedding by frame number')

  xlen = max(embedded_data[:, 0]) - min(embedded_data[:, 0])
  ylen = max(embedded_data[:, 1]) - min(embedded_data[:, 1])
  width = 0.03 * min(xlen, ylen)
  # Plot rectangles around keyframes.
  keyframe_points = embedded_data[keyframes]
  for p in keyframe_points:
    starting_corner = (p[0] - 0.5 * width, p[1] - 0.5 * width)
    rectangle = patches.Rectangle(starting_corner, width, width, fill=False, color='r')
    ax2.add_patch(rectangle)

  # Plot every nth frame as a baseline.
  # nth_frame_idx = np.arange(0, len(keyframes)) * int(len(embedded_data) / len(keyframes))
  # nth_frame_points = embedded_data[nth_frame_idx]
  # for p in nth_frame_points:
  #   circle = patches.Circle((p[0], p[1]), width * 0.5, fill=False, color='m')
  #   ax2.add_patch(circle)

  # Set up first subplot for viz.
  ax1.imshow(np.zeros((28, 28, 3)))
  embed_to_data = {(x[0], x[1]): data[i] for i, x in enumerate(embedded_data)}
  def redraw_data(event):
    ind = event.ind
    xy = event.artist.get_offsets()
    nearby_points = xy[ind]
    mousepoint = np.array([event.mouseevent.xdata, event.mouseevent.ydata])
    nearest_point_idx = np.argmin(np.linalg.norm(nearby_points - mousepoint))
    nearest_point = nearby_points[nearest_point_idx]
    hovered_data = embed_to_data[(nearest_point[0], nearest_point[1])]
    ax1.set_title('Frame No. %d' % ind[nearest_point_idx])
    ax1.imshow(hovered_data)
    fig.canvas.draw()

  fig.canvas.mpl_connect('pick_event', redraw_data)
  plt.show()

def plot_tsne_with_keyframes(embedded_data, data, keyframes, train_frames=None):
  fig, (ax1, ax2) = plt.subplots(1, 2)

  sampling_offset = 1
  sampled_embedded_data = embedded_data[::sampling_offset]
  # Plot data in both subplots.
  cax = ax1.scatter(sampled_embedded_data[:, 0], sampled_embedded_data[:, 1],
                    c=[float(i) for i in range(len(sampled_embedded_data))], 
                    picker=True, alpha=0.3, edgecolors='')
  ax1.set_title('t-SNE embedding with every nth frame')

  cax = ax2.scatter(sampled_embedded_data[:, 0], sampled_embedded_data[:, 1],
                    c=[float(i) for i in range(len(sampled_embedded_data))],
                    picker=True, alpha=0.3, edgecolors='')
  ax2.set_title('t-SNE embedding with keyframes')

  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  cbar = fig.colorbar(cax, cax=cbar_ax)

  xlen = max(embedded_data[:, 0]) - min(embedded_data[:, 0])
  ylen = max(embedded_data[:, 1]) - min(embedded_data[:, 1])
  width = 0.03 * min(xlen, ylen)
  # Plot every nth frame as a baseline.
  nth_frame_idx = np.arange(0, len(keyframes)) * int(len(embedded_data) / len(keyframes))
  nth_frame_points = embedded_data[nth_frame_idx]
  for p in nth_frame_points:
    circle = patches.Circle((p[0], p[1]), width * 0.5, fill=False, color='r')
    ax1.add_patch(circle)

  # Plot rectangles around keyframes.
  keyframe_points = embedded_data[keyframes]
  for p in keyframe_points:
    starting_corner = (p[0] - 0.5 * width, p[1] - 0.5 * width)
    rectangle = patches.Rectangle(starting_corner, width, width, fill=False, color='r')
    ax2.add_patch(rectangle)

  def plot_bounding_lines(ax, text, train_start_coord, train_end_coord):
    text_coord = (train_start_coord[0] - width * 20, 0.5 * (train_start_coord[1] + train_end_coord[1]))
    ax.text(text_coord[0], text_coord[1], text)
    ax.plot([train_start_coord[0] - width * 10, train_start_coord[0] + width * 10],
            [train_start_coord[1]] * 2, c='k')
    ax.plot([train_end_coord[0] - width * 10, train_end_coord[0] + width * 10],
            [train_end_coord[1]] * 2, c='k')

  if train_frames is not None:
    for i, train_frame in enumerate(train_frames):
      train_note = 'Train no. %d' % (i + 1)
      train_start, train_end = train_frame
      train_start_coord, train_end_coord = embedded_data[train_start], embedded_data[train_end]
      plot_bounding_lines(ax1, train_note, train_start_coord, train_end_coord)
      plot_bounding_lines(ax2, train_note, train_start_coord, train_end_coord)

  plt.show()

if __name__ == '__main__':
  data, labels = load_mnist()

  # subsample data
  idx = np.random.choice(len(data), size=1000, replace=False)
  data, labels = data[idx, :], labels[idx]
  rgb_data = np.concatenate([data.reshape(-1, 28, 28, 1)] * 3, axis=-1)

  embedded_data = apply_tsne(data)
  # plot_tsne(embedded_data, labels)
  plot_tsne_with_mouseover_viz(embedded_data, labels, rgb_data)
