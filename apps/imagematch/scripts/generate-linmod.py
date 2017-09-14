
import sys
import tensorflow as tf

def main():
  if(len(sys.argv) < 3 or "-h" in sys.argv[1]):
    print("Generates a linear model for imagematch for a particular vishash");
    print("Usage: show-layers.py vishash_size /path/to/output/model.pb");
    print("vishash_size: size of vishash (product of dimensions of tensor (excluding batch))");
    return;
  size_vishash = int(sys.argv[1]);
  output_path = sys.argv[2];
  output_folder, output_filename = output_path.rsplit('/', 1);
  # construct graph
  init_var = [];
  for i in range(size_vishash):
    init_var.append(0.01);

  x = tf.placeholder(dtype=tf.float32, shape=(size_vishash), name='x');
  expected = tf.placeholder(dtype=tf.float32, shape=(1), name='expected');
  # vector A (Ax + b)
  var = tf.Variable(init_var, name='var', dtype=tf.float32);
  skew = tf.Variable(0, name='skew', dtype=tf.float32);
  actual = tf.add(tf.reduce_sum(tf.multiply(var, x)), skew, name='actual');
  # difference squared
  loss = tf.square(tf.subtract(expected, actual), name='loss');
  optimizer = tf.train.AdamOptimizer(1e-3);
  train = optimizer.minimize(loss, name='train');
  init = tf.global_variables_initializer();

  # Write graph
  with tf.Session() as session:
    tf.train.write_graph(session.graph_def, output_folder, output_filename, as_text=False);
        
if(__name__ == "__main__"):
  main();
