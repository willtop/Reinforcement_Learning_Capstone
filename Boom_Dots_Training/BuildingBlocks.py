import numpy as np
import tensorflow as tf
#Don't need processing score: from input_processor import ScoreCalc

#SCORES
#waiting_reward = -0.2 # It was -0.5 however it would encourage agent to blindly make first tap
#lasting_reward = 0.2
terminate_reward = -1
#first_tab_reward = 0.5
correct_tab_reward = 0.6 # motivate such taps that keep agent alive
#negative_opposite_action_reward = 0 # not using this entry (used to be -1)


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
    #toAppend = []
    #actionAppend = []
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    tap = np.zeros(2)
    tap[1] = 1
    
    score = 0
    
    # print(len(data))
    # print(data.shape)
    # print(self.state.shape)
    
    for i in range(np.shape(data)[0]): 
      # due to 2 timesteps delay, the last 2 timesteps will be with agnostic terminal status, so not using them
      if(i == len(data)-1 or i == len(data)-2):
          toDelete.append(i)
          continue
      
      current_transac = data[i]
      # check if this is a invalid state (state after actual terminal)
      next_transac = data[i+1]; second_next_transac = data[i+2]
      if(current_transac[4] or next_transac[4]):
          # this is a transaction after reaching terminal state, due to 2 timesteps lag
          toDelete.append(i) # ignore this state
          continue
          
      # if this transaction reaches a terminal state  
      if (second_next_transac[4]):
          self.reward.append(terminate_reward)
          continue
             
      # upon reaching here, this state doesn't correspond to a terminal state
      if np.argmax(current_transac[1])==1:
          # special reward for making one right tap 
          reward = correct_tab_reward + self.discount*next_transac[4]
      else:
          # agent did nothing, don't have any corresponding reward (use time decay to implicitly reward it for staying active)
          reward = self.discount*next_transac[4]
      self.reward.append(reward)
      
    self.state = np.delete(self.state, toDelete)      
    self.action = np.delete(self.action, toDelete)      
    self.reward = np.array(self.reward)
    
    #vstack data to turn into ndarrays
    self.state = np.stack(self.state)
    self.action = np.stack(self.action)
    self.reward = np.stack(self.reward)
    
        
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