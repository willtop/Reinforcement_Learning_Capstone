import numpy as np
import tensorflow as tf

class DataDistribution:
	def __init__(self, dataset):
		self.train_images = dataset["inputs_train"]
		self.train_labels = dataset["targets_train"]
		self.val_images = dataset["inputs_val"]
		self.val_labels = dataset["targets_val"]
		assert self.train_images.shape[0] == self.train_labels.shape[0], (
					'images.shape: %s labels.shape: %s' % (self.train_images.shape, self.train_labels.shape))

		self._num_examples = self.train_images.shape[0]
		self._index_in_epoch = 0
		self._num_examples = len(self.train_images)
		self._epochs_completed = 0
		
		#shuffle data
		perm = np.arange(self._num_examples)
		np.random.shuffle(perm)
		self.train_images = self.train_images[perm]
		self.train_labels = self.train_labels[perm]
		
		print( self.train_images.shape)
		print( self.train_labels.shape)
		print( self.val_images.shape)
		print( self.val_labels.shape)
	
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
			self.train_images = self.train_images[perm]
			self.train_labels = self.train_labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = numSamples
			assert numSamples <= self._num_examples
		end = self._index_in_epoch
		return self.train_images[start:end], self.train_labels[start:end]
    

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)