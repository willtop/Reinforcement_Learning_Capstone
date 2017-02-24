import numpy as np
import tensorflow as tf
from input_processor import ScoreCalc

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
    ScoreCalculator = ScoreCalc()
    
    data = np.array(self.dataset)
    self.state = data[:,0]
    self.action = data[:,1]
    self.reward = []
    toDelete = []
    
    score = 0
    
    print(data.shape)
    print(self.state.shape)
    
    for i,d in enumerate(data):
      # No Q value for last data point
      if i == len(data):
        break
        
      # If we died, no reward
      if d[5]:
        self.reward.append(0)
        ScoreCalculator.resertScore()
        score = 0
        continue
        
      new_score = ScoreCalculator.getScore(d[3])
      #if score calculator failed, delete data point
      if new_score == -1:
        toDelete.append(i)
        continue
      #if score increased, then we succeed
      elif new_score > score:
        reward = 1 + self.discount*data[i+1][4]
        score = new_score
      #we did nothing this time
      else:
        reward = self.discount*data[i+1][4]
      self.reward.append(reward)
      
    self.state = np.delete(self.state, toDelete)      
    self.action = np.delete(self.action, toDelete)      
    self.reward = np.array(self.reward)
    
    assert self.state.shape[0] == self.reward.shape[0], (
          'state.shape: %s reward.shape: %s' % (self.state.shape, self.reward.shape))
    assert self.state.shape[0] == self.action.shape[0], (
          'state.shape: %s action.shape: %s' % (self.state.shape, self.action.shape))
          
    #vstack data to turn into ndarrays
    self.state = np.stack(self.state)
    self.action = np.stack(self.action)
    self.reward = np.stack(self.reward)

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