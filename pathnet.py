from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math 
import numpy as np

import tensorflow as tf 

class PathNet(object):
    def __init__(self, num_layers, num_modules, num_neurons_per_module):
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.num_neurons_per_module = num_neurons_per_module

    def variable_summaries(self, var):
        """
        Attach a lot of summaries to a tensor (Tensorboard visualization)
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.compat.v1.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
            tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
            tf.compat.v1.summary.histogram('histogram', var)

    def geopath_initializer(self):
        """
        Initialize the paths matrix
        """
        geopath = np.zeros((self.num_layers, self.num_modules), 
                            dtype=object)
        for i in range(self.num_layers):
            for j in range(self.num_modules):
                geopath[i,j] = tf.Variable(1.0)
        return geopath

    def module_weight_variable(self,shape):
        """
        Create a weight variable with appropriate initialization
        """
        initial = tf.random.truncated_normal(shape, stddev=0.1)
        return [tf.Variable(initial)]
    
    def module_bias_variable(self,shape):
        """
        Create a bias variable with appropriate initialization
        """
        initial = tf.constant(0.1, shape=shape)
        return [tf.Variable(initial)]

    def module(self, input_tensor, weights, biases, layer_name, act_func = tf.nn.relu):
        # Adding name scope to ensures logical grouping of the layers in the graph
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                self.variable_summaries(weights[0])
            
            with tf.name_scope('biases'):
                self.variable_summaries(biases[0])
            
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights[0]) + biases 
                tf.compat.v1.summary.histogram('pre_activations', preactivate)
            
            activations = act_func(preactivate, name='activation')
            tf.compat.v1.summary.histogram('activations', activations)
            return activations

    def module_hidden(self, i, input_tensor, weights, biases, layer_name, act_func = tf.nn.relu):
        # Adding name scope to ensures logical grouping of the layers in the graph 
        with tf.name_scope(layer_name):
            # skip layer
            if (i % 3 == 0):
                return input_tensor
            elif (i % 3 == 1):
                # Hold the state of the wieghts for the layer
                with tf.name_scope('weights'):
                    self.variable_summaries(weights[0])
                
                with tf.name_scope('biases'):
                    self.variable_summaries(biases[0])
                
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, weights[0]) + biases
                
                activations = act_func(preactivate, name='activation')
                tf.compat.v1.summary.histogram('activations', activations)
                return activations
            
            elif (i % 3 == 2):
                # Hold the state of the wieghts for the layer
                with tf.name_scope('weights'):
                    self.variable_summaries(weights[0])
                
                with tf.name_scope('biases'):
                    self.variable_summaries(biases[0])
                
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, weights[0]) + biases
                
                activations = act_func(preactivate, name='activation') + input_tensor
                tf.compat.v1.summary.histogram('activations', activations)
                return activations