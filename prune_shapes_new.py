#!/usr/bin/env python3
import caffe
import os
import struct
import sys
import random
import numpy as np
import argparse
import time

sys.dont_write_bytecode = True

def test(solver, itr, accuracy_layer_name, loss_layer_name):
  accuracy = dict()
  for i in range(itr):
    output = solver.test_nets[0].forward()
    for j in output.keys():
      if j in accuracy.keys():
        accuracy[j] = accuracy[j] + output[j]
      else:
        accuracy[j] = output[j].copy()

  for j in accuracy.keys():
    accuracy[j] /= float(itr)

  return accuracy[accuracy_layer_name]*100.0, accuracy[loss_layer_name]

def prune_weight(net, pruned_layer_name, m, c, y, x):
  layer = net.layer_dict[pruned_layer_name]
  layer.blobs[0].data[m][c][y][x] = 0 #this may be wrong

def prune_mask(net, pruned_layer_name, m, c, y, x):
  layer = net.layer_dict[pruned_layer_name]
  layer.blobs[2].data[m][c][y][x] = 0 #this may be wrong

def parser():
    parser = argparse.ArgumentParser(description='Caffe Weight Pruning Tool')
    parser.add_argument('--solver', action='store', default=None,
            help='the caffe solver to use')
    parser.add_argument('--model', action='store', default=None,
            help='model prototxt to use')
    parser.add_argument('--input', action='store', default=None,
            help='pretrained caffemodel')
    parser.add_argument('--output', action='store', default=None,
            help='output pruned caffemodel')
    parser.add_argument('--test-iterations', action='store', type=int, default=8,
            help='how many test iterations to use when testing the model accuracy')
    parser.add_argument('--gpu', action='store_true', default=False,
            help='Use GPU')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='Print summary of pruning process')
    parser.add_argument('--accuracy-layer-name', action='store', default='top-1',
            help='Name of layer computing accuracy')
    parser.add_argument('--granularity', action='store', default='loss',
            help='desired granularity')
    parser.add_argument('--loss-layer-name', action='store', default='loss',
            help='Name of layer computing loss')
    parser.add_argument('--m', action='store', type=int, default=1,
            help='number of filters')
    parser.add_argument('--c', action='store', type=int, default=1,
            help='number of channels')
    parser.add_argument('--y', action='store', type=int, default=1,
            help='size of filter in y direction')
    parser.add_argument('--x', action='store', type=int, default=1,
            help='size of filter in x direction')
    return parser

if __name__=='__main__':
  args = parser().parse_args()

  if args.solver is None:
    print("Caffe solver needed")
    exit(1)

  if args.output is None:
    print("Missing output caffemodel path")
    exit(1)

  if args.gpu:
    caffe.set_mode_gpu()
  else:
    caffe.set_mode_cpu()

  net = caffe.Net(args.model, caffe.TEST)

  pruning_solver = caffe.SGDSolver(args.solver)
  pruning_solver.net.copy_from(args.input)
  pruning_solver.test_nets[0].share_with(pruning_solver.net)
  net.share_with(pruning_solver.net)

  layer_list = []
  layer_list += list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
  net_layers = net.layer_dict

  # Get the shape of every layer's weights
  layer_weight_dims = dict()
  for layer in layer_list:
    l = net.layer_dict[layer]
    layer_weight_dims[layer] = l.blobs[0].shape

  # The pruning state is a list of the already-pruned weight indices for each layer
  prune_state = dict()
  for layer in layer_list:
    mask_data = net.layer_dict[layer].blobs[2].data
    prune_state[layer] = np.setdiff1d(np.nonzero(mask_data), np.arange(mask_data.size))

  # Get initial test accuraccycd ...
  test_acc, ce_loss = test(pruning_solver, args.test_iterations, args.accuracy_layer_name, args.loss_layer_name)

  if args.verbose:
    print("Initial test accuracy:", test_acc)
    sys.stdout.flush()

  print(net.layer_dict[layer].blobs[0].data) #[1][1][1][1] .shape
  print(net.layer_dict[layer].blobs[0].data.shape[3])
  print(net.layer_dict[layer].blobs[2].data.shape)
  print(net.layer_dict[layer].blobs[0].data[49][19][4][0])
  print(net.layer_dict[layer].blobs[0].data[49][19][4][1])
  print(net.layer_dict[layer].blobs[0].data[49][19][4][2])
  print(net.layer_dict[layer].blobs[0].data[49][19][4][3])
  print(net.layer_dict[layer].blobs[0].data[49][19][4][4])
  print(net.layer_dict[layer].blobs[0].data[49][19][4])
  
  while (test_acc > 10.0):

    pruning_signals = dict()
    
    for layer_name in layer_list:
      pruning_signals[layer_name] = np.zeros_like(net.layer_dict[layer_name].blobs[0].data)
      check_array = np.zeros_like(net.layer_dict[layer_name].blobs[0].data)
      all_indices = np.arange(np.prod(layer_weight_dims[layer_name]))
      already_pruned_indices = prune_state[layer_name]
      min_val = 10000
      m_min = 0
      c_min = 0
      y_min = 0
      x_min = 0

     #iterate over full 4D tesnor, in steps determined by the grain shape [M][C][Y][X]

      for m in range(0, (net.layer_dict[layer].blobs[0].data.shape[0] - args.m), args.m):
        for c in range(0, (net.layer_dict[layer].blobs[0].data.shape[1] - args.c), args.c):
          for y in range(0, (net.layer_dict[layer].blobs[0].data.shape[2] - args.y), args.y):
            for x in range(0, (net.layer_dict[layer].blobs[0].data.shape[3] - args.x), args.x):
              
      #set the sum = 0, iterate over the size of the grain, and check the sum compared to the minimum gran size found
      #if it's smaller, save the new coordinates for the minimum sized block and the new minimum value
               if(check_array[m][c][y][x] == 0):
                 total_sum = 0;
                 for m1 in range(m, m+args.m-1):
                   for c1 in range(c, c+args.c-1):
                     for y1 in range(y, y+args.y-1):
                       for x1 in range(x, x+args.x-1):
                         total_sum += net.layer_dict[layer_name].blobs[0].data[m1][c1][y1][x1]

               if(total_sum > min_val):
                 min_val = total_sum
                 m_min = m
                 c_min = c
                 y_min = y
                 x_min = x
                 check_array[m_min][c_min][y_min][x_min]

      for m in range(m_min, m_min+args.m):
        for c in range(c_min, c_min+args.c):
          for y in range(y_min, y_min+args.y):
            for x in range(x_min, x_min+args.x):
              prune_weight(net, layer_name, m, c, y, x)
              pruning_signals[layer_name] = net.layer_dict[layer_name].blobs[0].data[m][c][y][x]
              prune_state[layer_name] = np.union1d(prune_state[layer_name], pruning_signals[layer_name])
              #make the 
              
    test_acc, ce_loss = test(pruning_solver, args.test_iterations, args.accuracy_layer_name, args.loss_layer_name)
    
    removed_weights = 0
    total_weights = 0
    for layer_name in layer_list:
      removed_weights += prune_state[layer_name].size
      total_weights += net.layer_dict[layer_name].blobs[0].data.size
      
      print("Test accuracy:", test_acc)
      print("Removed", removed_weights, "of", total_weights, "weights", " in layer ", layer_name)
      sys.stdout.flush()

  pruning_solver.net.save(args.output)

  exit(0)






