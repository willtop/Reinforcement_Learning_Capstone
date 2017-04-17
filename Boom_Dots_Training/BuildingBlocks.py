import numpy as np
import tensorflow as tf
#Don't need processing score: from input_processor import ScoreCalc

#SCORES
waiting_reward = -0.2 # It was -0.5 however it would encourage agent to blindly make first tap
lasting_reward = 0.2
terminate_reward = -1
first_tab_reward = 0.5
correct_tab_reward = 0.5 # motivate such taps that keep agent alive
negative_opposite_action_reward = 0 # not using this entry (used to be -1)


class DataDistribution:
  def __init__(self, dataset, discountFactor):
    self.dataset = dataset
    self.discount = discountFactor
    # self.current_state = dataset["inputs_train"]
    # self.train_labels = dataset["targets_train"]
    # self.val_images = dataset["inputs_val"]
    # self.val_labels = dataset["targets_val"]
    # assert self.train_images.shape[0] == self.train_labels.shape[0], (
          # 'images.shape: %s labels.shape: %s' % (self.train_images.shape, self.train_labels.shape))

    # self._num_examples = self.train_images.shape[0]
    # self._index_in_epoch = 0
    # self._num_examples = len(self.train_images)
    # self._epochs_completed = 0
    
    # #shuffle data
    # perm = np.arange(self._num_examples)
    # np.random.shuffle(perm)
    # self.train_images = self.train_images[perm]
    # self.train_labels = self.train_labels[perm]
    
    # print( self.train_images.shape)
    # print( self.train_labels.shape)
    # print( self.val_images.shape)
    # print( self.val_labels.shape)
    

    
  def processInput(self):
    print("pre-processing")
    #Don't need score processor: ScoreCalculator = ScoreCalc()
    
    data = np.array(self.dataset)
    self.state = data[:,0]
    self.action = data[:,1]
    self.reward = []
    toDelete = []
    toAppend = []
    actionAppend = []
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    tap = np.zeros(2)
    tap[1] = 1
    
    score = 0
    
    # print(len(data))
    # print(data.shape)
    # print(self.state.shape)
    
    first_tab_performed = False
    for i,d in enumerate(data):
      # No Q value for last data point
      # print("Loop {}".format(i))
      if i == len(data)-1:
        toDelete.append(i)
        break
      
      # if this transaction reaches a terminal state  
      if d[5]:
        self.reward.append(terminate_reward)
        toAppend.append(d[0])
        # Q: Couldn't just append the action d[1]?
        if d[1][0] == 1:
          actionAppend.append(tap)
        elif d[1][1] == 1:
          actionAppend.append(do_nothing)
        else:
          print("ERROR: action was not tap nor do_nothing")
          exit()
        # have to manually reset the global variable for ongoing score to zero
        #ScoreCalculator.resertScore()
        #score = 0
        first_tab_performed = False
        continue
        
      increment = d[3]
      if increment:
          if not first_tab_performed: 
              # special case for just making that first tab
              reward = first_tab_reward + self.discount*data[i+1][4]
              first_tab_performed = True
              self.reward.append(reward)
              continue
          first_tab_performed = True

      if not first_tab_performed:
          # penalty for waiting to encourage performing first tab
          reward = waiting_reward + self.discount*data[i+1][4]
      else: # has performed first tab, good job for staying active
          reward = lasting_reward + self.discount*data[i+1][4]
          # special reward for making one right tap
          if(np.argmax(d[1])==1):
              reward += correct_tab_reward
      self.reward.append(reward)
      
    self.state = np.delete(self.state, toDelete)      
    self.action = np.delete(self.action, toDelete)      
    self.reward = np.array(self.reward)
    
    #vstack data to turn into ndarrays
    self.state = np.stack(self.state)
    self.action = np.stack(self.action)
    self.reward = np.stack(self.reward)
    
    if negative_opposite_action_reward != 0:
      toAppend = np.array(toAppend)
      actionAppend = np.array(actionAppend)
      rewardAppend = np.ones((len(toAppend)))*negative_opposite_action_reward
      # print(rewardAppend)
      # print(self.reward)
     
      self.state = np.vstack((self.state, toAppend))
      self.action = np.vstack((self.action, actionAppend))
      self.reward = np.concatenate((self.reward, rewardAppend))
        
    assert self.state.shape[0] == self.reward.shape[0], (
          'state.shape: %s reward.shape: %s' % (self.state.shape, self.reward.shape))
    assert self.state.shape[0] == self.action.shape[0], (
          'state.shape: %s action.shape: %s' % (self.state.shape, self.action.shape))
          
    self._num_examples = self.state.shape[0]
    self._index_in_epoch = 0
    self._epochs_completed = 0
    
    #shuffle data
    perm = np.arange(self._num_examples)
    np.random.shuffle(perm)
    self.state = self.state[perm]
    self.action = self.action[perm]
    self.reward = self.reward[perm]
    
    print(self.state.shape)
    print(self.action.shape)
    print(self.reward.shape)
    
    print("pre-processing done. Produced processed transactions with reward signals")

    
    # assert self.action[0].shape == (2)
    # stateshape = self.state[0].shape
    # for i in self.state:
      # assert i.shape == stateshape, ("Why is it broken?")
  
  def sample(self, numSamples):
    """
    Copied from tensorflow's mnist exmaple
    """
    start = self._index_in_epoch
    self._index_in_epoch += numSamples
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self.state = self.state[perm]
      self.action = self.action[perm]
      self.reward = self.reward[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = numSamples
      assert numSamples <= self._num_examples
    end = self._index_in_epoch
    return self.state[start:end], self.action[start:end], self.reward[start:end], self._epochs_completed
    

# def variable_summaries(var, name):
  # """Attach a lot of summaries to a Tensor."""
  # with tf.name_scope('summaries'):
    # mean = tf.reduce_mean(var)
    # tf.summary.scalar('mean/' + name, mean)
    # with tf.name_scope('stddev'):
      # stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    # tf.summary.scalar('stddev/' + name, stddev)
    # tf.summary.scalar('max/' + name, tf.reduce_max(var))
    # tf.summary.scalar('min/' + name, tf.reduce_min(var))
    # tf.summary.histogram(name, var)