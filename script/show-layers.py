import sys

import numpy as np
import tensorflow as tf
import google
from google.protobuf import text_format

FLAGS = tf.app.flags.FLAGS;

def main():
  # TODO: Does python guarantee that these expressions are evaluated from left to right?
  if(len(sys.argv) < 2 or "-h" in sys.argv[1]):
    print("Prints the names and sizes of all tensors in graph");
    print("Usage: show-layers.py /path/to/model.pb");
    return;

  model_path = sys.argv[1];

  # Import graph

  with tf.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.GraphDef();
  
    if(model_path[-5:] == "pbtxt"):
      graph_def = text_format.Parse(f.read(), tf.GraphDef())
    else:
      graph_def.ParseFromString(f.read());
    tf.import_graph_def(graph_def, name="", input_map=None, producer_op_list=None, op_dict=None, return_elements=None);

  with tf.Session() as sess:
    for op in sess.graph.get_operations():
      try:
        tensor = tf.Graph.get_tensor_by_name(tf.get_default_graph(), op.name + ":0");
      except:
        continue
      print("NAME = \"" + tensor.name + "\"  SHAPE = " + str(tensor.get_shape()));
    return;

main();
