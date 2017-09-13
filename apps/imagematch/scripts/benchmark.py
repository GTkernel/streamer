from __future__ import print_function;

import argparse;

import subprocess;
import sys;
import os;

import time;

import signal;
import numpy;

import multiprocessing

#sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
def set_cpu_governor():
  num_cpu = multiprocessing.cpu_count()
  cmd_base = ['echo', 'performance', '|', 'sudo', 'tee'];
  for i in range(0, num_cpu):
    cpu_file = "/sys/devices/system/cpu/cpu" + str(i) + "/cpufreq/scaling_governor";
    cmd = cmd_base.append(cpu_file);
    subprocess.Popen(cmd);

def main():
  parser = argparse.ArgumentParser(description="Run ImageMatch experiments.");
  parser.add_argument("--verbose", dest="verbose", action="store_true",
                       default=False, help="Verbose output");
  parser.add_argument("--trials", dest="trials",
                       help="Number of trials per experiment");
  parser.add_argument("--experiment", dest="experiment_filename",
                       help="Experiment file containing experiment parameters");
  parser.add_argument("--warmup", dest="warmup_iters", default=100,
                       help="Number of iterations to ignore (warmup)");
  args = parser.parse_args();
  trials = args.trials;
  warmup_iters = args.warmup_iters;
  cmd =  ['/home/tskim/streamer/build/apps/imagematch',
          '-c', 'GST_TEST',
          '-m', 'mobilenet',
          '-C', '/home/tskim/streamer/config/',
          '-v', 'prob',
          #'-q', '/home/tskim/models/input.jpg']
          '-q', 'fake']
  cmd.append('--use_fake_nne');
  experiment_file = open(args.experiment_filename, "r");
  DEVNULL = open(os.devnull, 'w');

  result_filename, extension = os.path.splitext(args.experiment_filename);
  outfile = open(result_filename + ".csv", "w", 0);
  header = "";
  for line in experiment_file:
    line = line.rstrip();
    new_cmd = cmd + line.split(' ');
    process = subprocess.Popen(new_cmd,
                               stderr=DEVNULL,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE
                              );
    out = "";
    cur_trial_cols = list();
    while(not("NeuralNetInference" in out)):
      out = process.stdout.readline().rstrip();

    if(header == ""):
      header = out;
      header_arr = header.split(',');
      new_header_str = "";
      for element in header_arr:
        new_header_str += element + ",";
        new_header_str += element + "-stdev,";
      print(new_header_str);
      print(new_header_str, file=outfile);

    if(len(cur_trial_cols) == 0):
      for i in range(0, header.count(',')):
        cur_trial_cols.append(list());

    for i in range(0, int(warmup_iters)):
      out = process.stdout.readline().rstrip();
      print("WARMUP: " + str(out));
    for i in range(0, int(trials)):
      out = process.stdout.readline().rstrip();
      out_split = out.split(",");
      out_split = out_split[:-1];
      for idx in range(0, len(out_split)):
        cur_trial_cols[idx].append(float(out_split[idx]));
      print(str(out) + "    (" + str(i) + "/" + trials + ") " + line);
    process.communicate("\n");
    output_str = "";
    for cur_trials in cur_trial_cols:
      avg = sum(cur_trials) / float(len(cur_trials));
      narr = numpy.array(cur_trials);
      stdev = numpy.std(narr);
      output_str += str(avg) + "," + str(stdev) + ",";
    print(output_str, file=outfile);
    

if __name__ == "__main__":
  main();
