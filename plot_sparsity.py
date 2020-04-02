#!/usr/bin/env python3

import numpy as np
import csv

#sys.dont_write_bytecode = True

#def parser():
#	parser = argparse.ArgumentParser(description='Plotting Sparsity/Accuracy')
#	parser.add_argument('--file', action='store', default=None, help='the file to plot')
#	parser.add_argument('--weights', action='store', default=None, help='number of weights in #		net')
#	return parser


if __name__=='__main__':
#	args = parser().parse_args()
#
#	if args.file is None:
#		print("Error loading file.")
#		exit(1)
#
#	weights_conv1 = 0
#	weights_conv2 = 0
#	total_weights = weights_conv1+weights_conv2
#	sparsity = 

#	file = args.file
	print("test1")

	with open('spars_15_3.txt', newline = '') as f:
		reader = csv.reader(f)
		for row in reader: 
			print("test2")
			print(row)
	print("test3")
	
#exit(0)
