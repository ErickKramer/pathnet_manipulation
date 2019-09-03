# Original implementation https://github.com/jaesik817/pathnet/blob/master/binary_mnist_pathnet.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import time 

import tensorflow as tf 
import utils
from pathnet import PathNet 

# Remove warnings from old version of dataset
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None 
def train():
    # Load Mnist data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True)
    # Remove warnings 
    tf.logging.set_verbosity(old_v)
    total_tr_data, total_tr_labels = mnist.train.next_batch(mnist.train._num_examples)
    # print(total_tr_data.shape)

    # Gathering images for class 1 of task 1
    tr_data_a1 = total_tr_data[(total_tr_labels[:,FLAGS.a1]==1.0)] 
    # Add noise to the images 
    tr_data_a1 = utils.add_noise(tr_data_a1)
    
    # Gathering images for class 2 of task 1
    tr_data_a2 = total_tr_data[(total_tr_labels[:,FLAGS.a2]==1.0)] 
    # Add noise to the images 
    tr_data_a2 = utils.add_noise(tr_data_a2)

    # Gathering images for class 1 of task 2 
    tr_data_b1 = total_tr_data[(total_tr_labels[:,FLAGS.b2]==1.0)] 
    # Add noise to the images 
    tr_data_b1 = utils.add_noise(tr_data_b1)

    # Gathering images for class 2 of task 2
    tr_data_b2 = total_tr_data[(total_tr_labels[:,FLAGS.b2]==1.0)] 
    # Add noise to the images 
    tr_data_b2 = utils.add_noise(tr_data_b2)
    
    # Grouping the classes of task 1
    tr_data_t1 = np.append(tr_data_a1, tr_data_a2, axis=0)
    # Set the labels of the two classes
    tr_label_t1 = np.zeros((len(tr_data_t1),2), dtype=float)
    tr_label_t1[0:len(tr_data_a1),0] = 1.0
    tr_label_t1[len(tr_data_a1):,1] = 1.0

    # Grouping the classes of task 2
    tr_data_t2 = np.append(tr_data_b1, tr_data_b2, axis=0)
    # Set the labels of the two classes
    tr_label_t2 = np.zeros((len(tr_data_t2),2), dtype=float)
    tr_label_t2[0:len(tr_data_b1),0] = 1.0
    tr_label_t2[len(tr_data_b1):,1] = 1.0

    # ------------------- Task 1 ---------------------
    sess = tf.compat.v1.InteractiveSession()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.float32, [None, tr_data_t1.shape[1]], name='image-input')
        y_ = tf.compat.v1.placeholder(tf.float32, [None, tr_label_t1.shape[1]], name='label-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.compat.v1.summary.image('input', image_shaped_input, 2)   
    
    # Declare PathNet 
    pathnet = PathNet(FLAGS.num_layer, FLAGS.num_mod, FLAGS.num_mod_sel)
    # Internal paths examples
    geopath = pathnet.geopath_initializer()

    # Fixed weight lists 
    # NOTE: Why is this necessary?
    fixed_list = np.ones((FLAGS.num_layer, FLAGS.num_mod), 
                          dtype=str)
    fixed_list[fixed_list == '1'] = '0'

    # Hidden layers 
    weights = np.zeros((FLAGS.num_layer, FLAGS.num_mod), 
                        dtype=object)
    biases = np.zeros((FLAGS.num_layer, FLAGS.num_mod), 
                        dtype=object)
    for i in range(FLAGS.num_layer):
        for j in range(FLAGS.num_mod):
            if (i==0):
                weights[i,j] = \
                    pathnet.module_weight_variable([tr_data_t1.shape[1], FLAGS.num_filt])
            else:
                weights[i,j] = \
                    pathnet.module_weight_variable([FLAGS.num_filt, FLAGS.num_filt])

            biases[i,j] = pathnet.module_bias_variable([FLAGS.num_filt])

    for i in range(FLAGS.num_layer):
        layer_modules = np.zeros(FLAGS.num_mod, dtype=object)
        
        for j in range(FLAGS.num_mod):
            if (i == 0):
                layer_modules[j] = pathnet.module(x, 
                                                  weights[i,j],
                                                  biases[i,j],
                                                  'layer'+str(i+1)+\
                                                  '_'+\
                                                  str(j+1))*geopath[i,j]
            else:
                layer_modules[j] = pathnet.module_hidden(j,
                                                   net,
                                                   weights[i,j],
                                                   biases[i,j],
                                                  'layer'+str(i+1)+\
                                                  '_'+\
                                                  str(j+1))*geopath[i,j]
        net = np.sum(layer_modules)/FLAGS.num_mod
    
    output_weights = pathnet.module_weight_variable([FLAGS.num_filt, 2])
    output_biases = pathnet.module_bias_variable([2])
    y = pathnet.nn_layer(net, output_weights, output_biases, 'output_layer')

    # Cross-entropy
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

    # Learning variables
    var_list_to_learn = [] + output_weights + output_biases
    for i in range(FLAGS.num_layer):
        for j in range(FLAGS.num_mod):
            if (fixed_list[i,j] == '0'):
                var_list_to_learn += weights[i,j] + biases[i,j]
    
    # Gradient descent
    with tf.name_scope('train'):
        train_step = tf.compat.v1.train.GradientDescentOptimizer(FLAGS.learning_rate).\
            minimize(cross_entropy,var_list=var_list_to_learn)

    # Accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.compat.v1.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out 
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries/
    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
   
    # Initialize all global variables
    tf.compat.v1.global_variables_initializer().run()  

    # Generate random internal path
    geopath_set = np.zeros(FLAGS.num_cand, dtype=object)
    for i in range(FLAGS.num_cand):
        geopath_set[i] = pathnet.get_geopath(FLAGS.num_layer,
                                             FLAGS.num_mod,
                                             FLAGS.num_mod_sel)

    # Parameters placeholders and ops
    var_update_ops = np.zeros(len(var_list_to_learn), dtype=object)
    var_update_placeholders = np.zeros(len(var_list_to_learn), dtype=object)
    for i in range(len(var_list_to_learn)):
        var_update_placeholders[i] = tf.compat.v1.placeholder(var_list_to_learn[i].dtype,
                                                              shape=var_list_to_learn[i].get_shape())
        var_update_ops[i] = var_list_to_learn[i].assign(var_update_placeholders[i])

    # Geopath placeholders and ops
    geopath_update_ops = np.zeros((len(geopath), len(geopath[0])), dtype=object)
    geopath_update_placeholders = np.zeros((len(geopath), len(geopath[0])), dtype=object)
    for i in range(len(geopath)):
        for j in range(len(geopath[0])):
            geopath_update_placeholders[i,j] = tf.compat.v1.placeholder(geopath[i,j].dtype,
                                                                        shape=geopath[i,j].get_shape())
            geopath_update_ops[i,j] = geopath[i,j].assign(geopath_update_placeholders[i,j])
    
    acc_geo = np.zeros(FLAGS.num_comp, dtype=float)
    summary_geo = np.zeros(FLAGS.num_comp, dtype=object)
    # Iterate over the training steps
    for i in range(FLAGS.max_steps):
        # Tournament selection
        ## Generate candidates
        candidates_idx = list(range(FLAGS.num_cand))
        ## Shuffle candidates 
        np.random.shuffle(candidates_idx)
    #     ## Choose candidates
        candidates_idx = candidates_idx[:FLAGS.num_comp]

        # Learning and evaluation
        ## Iterate over the selected candidates
        for j in range(len(candidates_idx)):
            # shuffle data 
            idx = list(range(len(tr_data_t1)))
            np.random.shuffle(idx)
            tr_data_t1 = tr_data_t1[idx]
            tr_label_t1 = tr_label_t1[idx]

            # Insert Candidate
            pathnet.geopath_insert(sess, 
                                   geopath_update_placeholders,
                                   geopath_update_ops,
                                   geopath_set[candidates_idx[j]],
                                   FLAGS.num_layer,
                                   FLAGS.num_mod)
            acc_geo_tr = 0
            for k in range(FLAGS.num_epoch_path):
                summary_geo_tr, _, acc_geo_tmp = sess.run([merged, train_step, accuracy],
                                                          feed_dict={x:tr_data_t1[k*FLAGS.num_batch:(k+1)*FLAGS.num_batch,:],
                                                                     y_:tr_label_t1[k*FLAGS.num_batch:(k+1)*FLAGS.num_batch,:]})
                acc_geo_tr += acc_geo_tmp
            acc_geo[j] = acc_geo_tr/FLAGS.num_epoch_path
            summary_geo[j] = summary_geo_tr

        # Tournament
        winner_idx = np.argmax(acc_geo)
        acc = acc_geo[winner_idx]
        summary = summary_geo[winner_idx]

        # Copy and mutation
        for j in range(len(candidates_idx)):
            if (j != winner_idx):
                geopath_set[candidates_idx[j]] = np.copy(geopath_set[candidates_idx[winner_idx]])
                geopath_set[candidates_idx[j]] = pathnet.mutation(geopath_set[candidates_idx[j]],
                                                                  FLAGS.num_layer,
                                                                  FLAGS.num_mod,
                                                                  FLAGS.num_mod_sel)
        train_writer.add_summary(summary,i) 
        print('Training Accuracy at step {}: {}'.format(i, acc))

        if (acc >= 0.99):
            print('Learning task complete')
            print('Optimal path is: ')
            task_1_optimal_path = geopath_set[candidates_idx[winner_idx]]
            print(task_1_optimal_path)
            break
        
    iter_task = i

    # Freeze task 1 optimal path
    var_list_to_fix = []
    for i in range(FLAGS.num_layer):
        for j in range(FLAGS.num_mod):
            if (task_1_optimal_path[i,j] == 1.0):
                fixed_list[i,j] = '1'
                var_list_to_fix += weights[i,j] + biases[i,j]
    
    # Get variables of fixed list
    var_list_fix = pathnet.parameters_backup(var_list_to_fix)

    # Parameters placeholders and ops
    var_fix_ops = np.zeros(len(var_list_to_fix), dtype=object)
    var_fix_placeholders = np.zeros(len(var_list_to_fix), dtype=object)
    for i in range(len(var_list_to_fix)):
        var_fix_placeholders[i] = tf.compat.v1.placeholder(var_list_to_fix[i].dtype,
                                                            shape=var_list_to_fix[i].get_shape())
        var_fix_ops[i] = var_list_to_fix[i].assign(var_fix_placeholders[i])

    # TODO: Create a generic function for a task 

    # ------------------- TASK 2 ------------------
    var_list_to_learn = [] + output_weights + output_biases
    for i in range(FLAGS.num_layer):
        for j in range(FLAGS.num_mod):
            if (fixed_list[i,j] == '0'):
                var_list_to_learn += weights[i,j] + biases[i,j]
            
    
    for i in range(FLAGS.num_layer):
        for j in range(FLAGS.num_mod):
            if (fixed_list[i,j] == '1'):
                tmp = biases[i,j][0]
                break
        break
    
    # Initialization
    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir +
                                                    '/train2', sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir +
                                                    '/test2', sess.graph)
    tf.compat.v1.global_variables_initializer().run()

    # Update fixed values
    pathnet.parameters_update(sess, var_fix_placeholders, var_fix_ops, var_list_fix)

    # GradientDescent
    with tf.name_scope('train'):
        train_step = tf.compat.v1.train.GradientDescentOptimizer(FLAGS.learning_rate).\
                        minimize(cross_entropy, var_list=var_list_to_learn)

    # Generating random geopath
    geopath_set = np.zeros(FLAGS.num_cand, dtype=object)
    for i in range(FLAGS.num_cand):
        geopath_set[i] = pathnet.get_geopath(FLAGS.num_layer, FLAGS.num_mod, FLAGS.num_mod_sel)
    
    # Parameters and placeholders and ops 
    var_update_ops = np.zeros(len(var_list_to_learn), dtype=object)
    var_update_placeholders = np.zeros(len(var_list_to_learn), dtype=object)
    for i in range(len(var_list_to_fix)):
        var_update_placeholders[i] = tf.compat.v1.placeholder(var_list_to_learn[i].dtype,
                                                            shape=var_list_to_learn[i].get_shape())
        var_update_ops[i] = var_list_to_learn[i].assign(var_update_placeholders[i])
    
    acc_geo = np.zeros(FLAGS.num_comp, dtype=float)
    summary_geo = np.zeros(FLAGS.num_comp, dtype=object)
    # Iterate over the training steps
    for i in range(FLAGS.max_steps):
        # Tournament selection
        ## Generate candidates
        candidates_idx = list(range(FLAGS.num_cand))
        ## Shuffle candidates 
        np.random.shuffle(candidates_idx)
    #     ## Choose candidates
        candidates_idx = candidates_idx[:FLAGS.num_comp]

        # Learning and evaluation
        ## Iterate over the selected candidates
        for j in range(len(candidates_idx)):
            # shuffle data 
            idx = list(range(len(tr_data_t2)))
            np.random.shuffle(idx)
            tr_data_t2 = tr_data_t2[idx]
            tr_label_t2 = tr_label_t2[idx]
            geopath_insert = np.copy(geopath_set[candidates_idx[j]])

            for l in range(FLAGS.num_layer):
                for m in range(FLAGs.num_mod):
                    if (fixed_list[l,m] == '1'):
                        geopath_insert[l,m] = 1.0
        
            # Insert Candidate
            pathnet.geopath_insert(sess, 
                                   geopath_update_placeholders,
                                   geopath_update_ops,
                                   geopath_insert,
                                   FLAGS.num_layer,
                                   FLAGS.num_mod)
            acc_geo_tr = 0
            for k in range(FLAGS.num_epoch_path):
                summary_geo_tr, _, acc_geo_tmp = sess.run([merged, train_step, accuracy],
                                                          feed_dict={x:tr_data_t2[k*FLAGS.num_batch:(k+1)*FLAGS.num_batch,:],
                                                                     y_:tr_label_t2[k*FLAGS.num_batch:(k+1)*FLAGS.num_batch,:]})
                acc_geo_tr += acc_geo_tmp
            acc_geo[j] = acc_geo_tr/FLAGS.num_epoch_path
            summary_geo[j] = summary_geo_tr

        # Tournament
        winner_idx = np.argmax(acc_geo)
        acc = acc_geo[winner_idx]
        summary = summary_geo[winner_idx]

        # Copy and mutation
        for j in range(len(candidates_idx)):
            if (j != winner_idx):
                geopath_set[candidates_idx[j]] = np.copy(geopath_set[candidates_idx[winner_idx]])
                geopath_set[candidates_idx[j]] = pathnet.mutation(geopath_set[compete_idx[j]],
                                                                  FLAGS.num_layer,
                                                                  FLAGS.num_mod,
                                                                  FLAGS.num_mod_sel)
        train_writer.add_summary(summary,i) 
        print('Training Accuracy at step {}: {}'.format(i, acc))

        if (acc >= 0.99):
            print('Learning task complete')
            print('Optimal path is: ')
            task_2_optimal_path = geopath_set[candidates_idx[winner_idx]]
            print(task_2_optimal_path)
            break

    iter_task2 = i

    overlap = 0
    for i in range(len(task_1_optimal_path)):
        for j in range(len(task_1_optimal_path[0])):
            if ((task_1_optimal_path[i,j] == task_2_optimal_path[i,j]) &
               (task_1_optimal_path[i,j] == 1.0)):
               overlap += 1
    print("Entire Iter: " + str(iter_task + iter_task2) + " , Task 1: " + 
          str(iter_task) + " ,Task 2: " + str(iter_task2) + " ,Overlap: " + str(overlap))

    train_writer.close()
    test_writer.close()
            



    

def main(_):
    FLAGS.log_dir += str(int(time.time()))
    
    if tf.io.gfile.exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.io.gfile.makedirs(FLAGS.log_dir)
    train()

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
    parser.add_argument('--num_mod_sel', type=int, default=3,
                        help='Number of selected modules per layer')
    parser.add_argument('--num_epoch_path', type=int, default=50,
                        help='Number of epochs per path')
    parser.add_argument('--num_batch', type=int, default=16,
                        help='Number of batches per path')
    parser.add_argument('--num_filt', type=int, default=20,
                        help='Number of filters per module')
    parser.add_argument('--num_cand', type=int, default=20,
                        help='Number of candidates per path')
    parser.add_argument('--num_comp', type=int, default=2,
                        help='Number of candidates per competition')
    parser.add_argument('--a1', type=int, default=1,
                        help='First class of task 1')
    parser.add_argument('--a2', type=int, default=1,
                        help='Second class of task 1')
    parser.add_argument('--b1', type=int, default=1,
                        help='First class of task 2')
    parser.add_argument('--b2', type=int, default=1,
                        help='Second class of task 2')

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed) 
