#!python3
"""
NEED TO IMPLEMENT:
game.GameState() object
with framestep function
"""

from game_params import *
import tensorflow as tf
import cv2

# import input_processor as game
import random
import time
import math
import numpy as np
from collections import deque
from BuildingBlocks import DataDistribution
from game_wrapper import Game

GAME = 'stack'  # the name of the game being played for log files
GAME_PARAMS = Jason_stack_params_phone
ACTIONS = 2  # number of valid actions
INPUT_DIMS = (60, 100)
NUM_FRAMES = 4 # Number of frames in each training data point
GAMMA = 0.90  # decay rate of past observations
# OBSERVE = 500.  # timesteps to observe before training
EXPLORE = 40  # iterations over which to anneal epsilon
FINAL_EPSILON = 4  # final value of epsilon
INITIAL_EPSILON = 0  # starting value of epsilon
# a value of 0 corresponds to random decision
REPLAY_MEMORY = 2000  # number of previous transitions to remember
# Total size of training data
TRAINING_ITER = 5 #Number of training iterations over the training data
BATCH = 32  # size of minibatch
LEARNING_RATE = 1e-3
RENDER_DISPLAY = True


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def create_network():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1792, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, INPUT_DIMS[0], INPUT_DIMS[1], NUM_FRAMES])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1792])
    # Should be 1792???

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    
    # print(h_conv1.get_shape())
    # print(h_pool1.get_shape())
    # print(h_conv2.get_shape())
    # print(h_conv3.get_shape())
    # print(h_conv3_flat.get_shape())

    return s, readout, h_fc1, train_step, a, y

def play_game(s, readout, h_fc1, sess, epsilon, restore = False):
    '''
    Plays the game and saves the training data to a python collections.
    It will fill up to REPLAY_MEMORY training points where each training
    point consists of:
      s_t: a game state (a sequence of NUM_FRAMES frames)
      r_t: a reward signal
      a_t: action taken after the last frame
      terminal: boolean indicating whether we lost or not
    '''
    # screenshot_dims = []
    # game_params={'restart_tap_position': (100,100)}

    # open up a game state to communicate with emulator
    game = Game(INPUT_DIMS, GAME_PARAMS, auto_restart=False)

    # store the previous observations in replay memory
    D = []

    # printing
    # log_file = open(GAME + "_output.txt", 'w')
    # h_file = open("logs_" + GAME + "/hidden.txt", 'w')
    
    if restore:
        restore_network()
    else:
        print("Did not restore network")

    # get the first state by doing nothing and preprocess the image to INPUT_DIMSx4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    tap = np.zeros(ACTIONS)
    tap[1] = 1
    
    game.frame_step(tap)
    time.sleep(0.1)

    x_t1 = [1,2,3,4]
    x_t_1, _, terminal = game.frame_step(do_nothing)
    x_t_2, _, terminal = game.frame_step(do_nothing)
    x_t_3, _, terminal = game.frame_step(do_nothing)
    x_t_4, _, terminal = game.frame_step(do_nothing)
    # print(x_t_1.shape)
    s_t = np.stack((x_t_1, x_t_2, x_t_3, x_t_4),axis=2)
    # print(s_t.shape)
    
    t = 0
    while t < REPLAY_MEMORY:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        Q_max_last = np.max(readout_t)
        # Choose action greedily
        if epsilon == -1:
          a_t_i = np.argmax(readout_t)
          a_t[a_t_i] = 1
          action_index = a_t_i
        else:
          tap_unnorm = np.exp(epsilon*readout_t[1])
          norm = np.sum(np.exp(epsilon*readout_t))
          Prob_Tap = tap_unnorm/norm
          decision = random.random()
          if decision <= Prob_Tap:
              a_t[1] = 1
              action_index = 1
          else:
              a_t[0] = 1
              action_index = 0
            
        # Apply action and get 3 next
        x_t1[0], _, terminal = game.frame_step(a_t)
        ### DEBUG ###
        # x_t1[0], _,  terminal = game.frame_step(do_nothing)
        ######
        for i in range(1, NUM_FRAMES):
            time.sleep(0.1)
            # Get next three
            x_t1[i], score, terminal = game.frame_step(do_nothing)
        s_t1 = np.stack((x_t1[0], x_t1[1], x_t1[2], x_t1[3]), axis=2)
            
        # store the transition in D
        D.append([s_t, a_t, s_t1, score, Q_max_last, terminal])
            
        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ Q_TAP %g" % readout_t[1], "/ Q_NONE %g" % readout_t[0], "/ TERMINAL ", terminal)
        # print("Prob_Tap {} / decision {}".format(Prob_Tap, decision))
        # print("Tap_unnorm {} / Tap_norm {} / Prob_Tap {} / decision {}".format(tap_unnorm, norm, Prob_Tap, decision))
        # log_file.write("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ Q_MAX %e" % Q_max_last, "/ TERMINAL ", terminal, '\n')
        
        if RENDER_DISPLAY:
            s_t_display = np.concatenate(np.rollaxis(s_t,2))
            s_t1_display = np.concatenate(np.rollaxis(s_t1,2))
            cv2.namedWindow('s_t', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('s_t', INPUT_DIMS[0] * 3 * NUM_FRAMES, INPUT_DIMS[1] * 3)
            cv2.imshow('s_t', cv2.cvtColor(np.transpose(s_t_display), cv2.COLOR_GRAY2BGR))
            cv2.namedWindow('s_t_1', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('s_t_1', INPUT_DIMS[0] * 3 * NUM_FRAMES, INPUT_DIMS[1] * 3)
            cv2.imshow('s_t_1', cv2.cvtColor(np.transpose(s_t1_display), cv2.COLOR_GRAY2BGR))
            cv2.namedWindow('score', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('score', (GAME_PARAMS.score_box[2]-GAME_PARAMS.score_box[0])*3, (GAME_PARAMS.score_box[3]-GAME_PARAMS.score_box[1])*3)
            cv2.imshow('score', cv2.cvtColor(score, cv2.COLOR_GRAY2BGR))
            cv2.waitKey(1)
        
        # update the old values
        s_t = s_t1
        t += 1
        
        if terminal:
          game.restart()
          x_t1 = [1,2,3,4]
          x_t_1, _, terminal = game.frame_step(do_nothing)
          x_t_2, _, terminal = game.frame_step(do_nothing)
          x_t_3, _, terminal = game.frame_step(do_nothing)
          x_t_4, _, terminal = game.frame_step(do_nothing)
          # print(x_t_1.shape)
          s_t = np.stack((x_t_1, x_t_2, x_t_3, x_t_4),axis=2)
          # input()
            
        
        
    # Store Data into a DataDistribution class and return
    
    game.reach_terminal_state()
    return D
    
def train_network(s, a, y, train_step, sess, data, iter, restore=False):
    '''
    Trains the network given the data saved from playing the game.
    '''
    saver = tf.train.Saver()
    if restore:
      restore_network()
    else:
        print("Did not restore network")
  
    i = 0
    # only train if done observing
    while i < TRAINING_ITER:
        # sample a minibatch to train on
        state, action, reward, i = data.sample(BATCH)
        # action = tf.reshape(action, [BATCH, ACTIONS])

        # assert all(x.shape == (INPUT_DIMS[0], INPUT_DIMS[1], NUM_FRAMES) for x in state)
        # assert all(x.shape == (ACTIONS) for x in action)
        # perform gradient step
        train_step.run(feed_dict={y: reward, a: action, s: state})
            
    saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=iter)
  
def restore_network():
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    s, readout, h_fc1, train_step, a, y = create_network()
    epsilon = INITIAL_EPSILON
    sess.run(tf.global_variables_initializer())
        
    # ### TRAINING SECTION ###
    # # For each loop, we save Test_Data_i.npy which is the collection of the 2k training examples
    # # and the model after training in saved_networks/
    
    # # If resuming training, set start accordingly
    # # start = 0 if there is not previous training data
    start = 43
    
    # for i in range(start):
      # # ## Run the below commented code if you want to retrain the model with the save play data
      # # data =  np.load("Test_Data_{}.npy".format(i))
      # # data = DataDistribution(data, GAMMA)
      # # data.processInput()
      # # train_network(s, a, y, train_step, sess, data, i)
      # if epsilon < FINAL_EPSILON:
          # epsilon += (FINAL_EPSILON - INITIAL_EPSILON) / EXPLORE
          
    # # ## Call restore_network() only if resuming trainig
    # restore_network()
    
    # for i in range(start,50):
      # start_time = time.time()
      # # play game gets the 2k training points
      # data = play_game(s, readout, h_fc1, sess, epsilon, restore=False)
      # np.save("Test_Data_{}".format(i), data)
      # play_time = time.time()
      
      # # DataDistribution and processInput calculate the state, reward pairs
      # data = DataDistribution(data, GAMMA)
      # data.processInput()
      # train_network(s, a, y, train_step, sess, data, i)
      # # scale down epsilon
      # if epsilon < FINAL_EPSILON:
          # epsilon += (FINAL_EPSILON - INITIAL_EPSILON) / EXPLORE
      # total_time = time.time()
      # print("Loop {}:".format(i))
      # print("Playing time took {}".format(play_time-start_time))
      # print("Training time took {}".format(total_time-play_time))
      
    ### TESTING SECTION ###
    # Epsilson should range from 0 (random) to +inf. Set to -1 for deterministic play
    epsilson = 5
    _ = play_game(s, readout, h_fc1, sess, epsilson, restore=True)
    
