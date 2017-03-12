import tensorflow as tf
import cv2
import sys
import time
import random
import numpy as np
import dqn as dqn
sys.path.append("wrapped_game_code/")
# import dummy_game as game
# import pong_fun as game
import android_game as game

GAME = game.NAME  # the name of the game being played for log files
ACTIONS = game.ACTIONS  # number of valid actions
GAMMA = game.GAMMA  # decay rate of past observations
OBSERVE = game.OBSERVE  # timesteps to observe before training
EXPLORE = game.EXPLORE  # frames over which to anneal epsilon
FINAL_EPSILON = game.FINAL_EPSILON  # final value of epsilon
INITIAL_EPSILON = game.INITIAL_EPSILON  # starting value of epsilon
REPLAY_MEMORY = game.REPLAY_MEMORY  # number of previous transitions to remember
REPLAY_MEMORY_DISCARD_AMOUNT = game.REPLAY_MEMORY_DISCARD_AMOUNT  # number of previous transitions to discard when replay memory is full
BATCH = game.BATCH  # size of minibatch

LEARNING_RATE = 1e-5
PLAY_TO_WIN = False
CHECKPOINTS_DIR = 'checkpoints_' + GAME + '/'


def train(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = []

    # printing
    # a_file = open("logs_" + GAME + "/readout.txt", 'w')
    # h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    t = 0

    checkpoint = tf.train.get_checkpoint_state('checkpoints_' + GAME)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        t = int(checkpoint.model_checkpoint_path.split('-')[-1])
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    while "pigs" != "fly":

        if PLAY_TO_WIN:
            epsilon = 0
        else:
            # scale down epsilon
            if t < OBSERVE:
                epsilon = INITIAL_EPSILON
            else:
                epsilon_delta_per_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                epsilon = INITIAL_EPSILON - epsilon_delta_per_step * (t - OBSERVE)
                epsilon = 0 if epsilon < 0 else epsilon

        # choose an action epsilon greedily
        readout_time = time.time()
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        readout_time = time.time() - readout_time

        # run the selected action and observe next state and reward
        frame_time = time.time()
        x_t1_col, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, 0:3], axis=2)
        frame_time = time.time() - frame_time

        train_time = time.time()
        if not PLAY_TO_WIN:
            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))

            # train on the entire replay memory when replay memory is full, then discard a portion of it
            if len(D) == REPLAY_MEMORY:
                indices = list(range(0, len(D)))
                random.shuffle(indices)

                minibatch = []
                for i in indices:
                    minibatch.append(D[i])
                    if len(minibatch) == BATCH:
                        # get the batch variables
                        s_j_batch = [d[0] for d in minibatch]
                        a_batch = [d[1] for d in minibatch]
                        r_batch = [d[2] for d in minibatch]
                        s_j1_batch = [d[3] for d in minibatch]

                        y_batch = []
                        readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                        for i in range(0, len(minibatch)):
                            # if terminal only equals reward
                            if minibatch[i][4]:
                                y_batch.append(r_batch[i])
                            else:
                                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                        # perform gradient step
                        train_step.run(feed_dict={
                            y: y_batch,
                            a: a_batch,
                            s: s_j_batch})

                        minibatch.clear()

                # empty out a portion of the replay memory
                D = D[REPLAY_MEMORY_DISCARD_AMOUNT:]
        train_time = time.time() - train_time

        # save progress every 10000 iterations
        if t % 10000 == 0 and t > 0 and not PLAY_TO_WIN:
            saver.save(sess, CHECKPOINTS_DIR + 'checkpoint', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if t % 100 == 0:
            print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", "{0:.3f}".format(epsilon), "/ ACTION", action_index,
                  "/ REWARD", "{0:5}".format(r_t), "/ Q_MAX %e" % np.max(readout_t), "/ TERMINAL", "True " if terminal else "False",
                  "/ READOUT_TIME", "{0:.4f}".format(readout_time),
                  "/ FRAME_TIME", "{0:.4f}".format(frame_time),
                  "/ BATCH_TIME", "{0:.4f}".format(train_time))

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

        # update the old values
        s_t = s_t1
        t += 1


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = dqn.create_network(ACTIONS)
    train(s, readout, h_fc1, sess)
