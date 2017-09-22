from tsne_utils import plot_tsne_with_mouseover_viz_no_label
import numpy as np
import sys

if __name__ == '__main__':
  data = np.load(sys.argv[1])
  embedded_data = np.load(sys.argv[2])
  plot_tsne_with_mouseover_viz_no_label(embedded_data, data)
