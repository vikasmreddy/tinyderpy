import sys
import argparse
from create_training_data import create_training_example_generator as cteg
from create_training_data import print_board_state as print_board_state
from create_training_data import *
import tensorflow as tf
import itertools
import json

class TinyDerpyModel:
    x = None
    W = None
    b = None
    y = None
    y_ = None
    train_step = None

def get_model():
    tdm = TinyDerpyModel()
    tdm.x = tf.placeholder(tf.float32, [None, 19])
    tdm.W = tf.Variable(tf.zeros([19, 9]))
    tdm.b = tf.Variable(tf.zeros([9]))
    tdm.y = tf.nn.softmax(tf.matmul(tdm.x, tdm.W) + tdm.b)
    tdm.y_ = tf.placeholder(tf.float32, [None, 9])

    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tdm.y_, logits=tdm.y))

    tdm.train_step = tf.train.GradientDescentOptimizer(3.0).minimize(cross_entropy)

    return tdm

def process_command_line_args():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('mode', choices=['save_training_data', 'use_saved_training_data', 'use_saved_model'])
    args = parser.parse_args()
    return args

def save_training_data_files(batch_xs_list, batch_ys_list):
    with open('batch_xs_list.json', 'w') as f:
        json.dump(batch_xs_list, f)

    with open('batch_ys_list.json', 'w') as f:
        json.dump(batch_ys_list, f)

def load_training_data_files():
    batch_xs_list = []
    batch_ys_list = []

    with open('batch_xs_list.json', 'r') as f:
        batch_xs_list = json.load(f)

    with open('batch_ys_list.json', 'r') as f:
        batch_ys_list = json.load(f)

    return batch_xs_list, batch_ys_list

def run_training(sess, saver, tdm, batch_xs_list, batch_ys_list):
    for i in range(0,1000):
        sess.run(tdm.train_step, feed_dict={tdm.x: batch_xs_list, tdm.y_: batch_ys_list})

    correct_prediction = tf.equal(tf.argmax(tdm.y,1), tf.argmax(tdm.y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={tdm.x: batch_xs_list, tdm.y_: batch_ys_list}))

    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)

def get_training_examples():
    training_example_generator = cteg()
    batch_xs_list = []
    batch_ys_list = []
    for example in training_example_generator:
        #print example
        batch_xs, batch_ys = example[0],example[1]
        batch_xs_list.append(batch_xs)
        batch_ys_list.append(batch_ys)

    return batch_xs_list, batch_ys_list

def run_testing(sess, tdm):
    prediction = tf.argmax(tdm.y,1)

    total_examples = 0
    total_correct = 0
    total_wrong = 0

    testing_example_generator = cteg()
    for example in testing_example_generator:
        batch_xs, batch_ys = example[0],example[1]
        board_state, whose_turn = vector_to_board_state_and_whose_turn(batch_xs)
        batch_xs = [batch_xs]
        batch_ys = [batch_ys]
        prediction_value = prediction.eval(feed_dict={tdm.x: batch_xs})
        predicted_move = regressed_value_to_move(prediction_value[0])
        correct_move = regressed_value_to_move(binary_vector_one_hot_to_move_to_index(example[1]))
        total_examples = total_examples + 1
        if predicted_move[0] == correct_move[0] and predicted_move[1] == correct_move[1]:
            total_correct = total_correct + 1
            print "correct move"
        else:
            total_wrong = total_wrong + 1
            print "wrong move"
            print predicted_move
            print correct_move

    print ("Total: {:d}, correct: {:d}, wrong: {:d}, accuracy: {:.3f}\n".format(
        total_examples,
        total_correct,
        total_wrong,
        total_correct/float(total_examples)))

def main_program():
    args = process_command_line_args()

    tdm = get_model()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    if args.mode == "save_training_data":
        batch_xs_list, batch_ys_list = get_training_examples()
        save_training_data_files(batch_xs_list, batch_ys_list)
        return
    else:
        if args.mode == "use_saved_model":
            saver.restore(sess, "model.ckpt")
        else:
            batch_xs_list, batch_ys_list = [], []
            if args.mode == "use_saved_training_data":
                batch_xs_list, batch_ys_list = load_training_data_files()
            else:
                batch_xs_list, batch_ys_list = get_training_examples()
            run_training(sess, saver, tdm, batch_xs_list, batch_ys_list)

    run_testing(sess, tdm)

main_program()
