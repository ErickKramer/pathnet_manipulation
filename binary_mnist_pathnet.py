# Original implementation https://github.com/jaesik817/pathnet/blob/master/binary_mnist_pathnet.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import time 

import tensorflow as tf 

FLAGS = None 

def main():

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Initial learning rate')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of steps to run trainer')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Probability of training dropout')
    parser.add_argument('--data_dir', type=str, default='data/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='logs/mnist',
                        help='Directory for storing logs')
    parser.add_argument('--num_mod', type=int, default=10,
                        help='Number of Modules per Layer')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--mod_sel', type=int, default=3,
                        help='Number of selected modules per layer')
    parser.add_argument('--epoch_path', type=int, default=50,
                        help='Number of epochs per path')
    parser.add_argument('--num_batch', type=int, default=16,
                        help='Number of batches per path')
    parser.add_argument('--num_filt', type=int, default=20,
                        help='Number of filters per module')
    parser.add_argument('--num_cand', type=int, default=20,
                        help='Number of candidates per path')
