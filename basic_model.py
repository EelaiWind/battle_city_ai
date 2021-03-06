import tensorflow as tf
import os
import sys

class BasicModel(object):
	def __init__(self, session, scope, checkpoint_path=None, weight_decay=0.0):
		self._session = session
		self._scope = scope
		self._weight_decay = weight_decay
		with tf.variable_scope(scope):
			self._learning_rate = tf.placeholder(tf.float32)
			self._build_model()
			self._trainable_variables = sorted([t for t in tf.trainable_variables() if t.name.startswith(self.scope)], key=lambda x: x.name)
			self._build_loss()
			if checkpoint_path is not None: self._setupSaver(checkpoint_path)

	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def scope(self):
		return self._scope

	@property
	def session(self):
		return self._session

	@property
	def trainable_variables(self):
		return self._trainable_variables

	def _build_loss(self):
		raise NotImplementedError("_build_loss() in %s" % __file__)

	def _build_model(self):
		raise NotImplementedError("_build_model() in %s" % __file__)

	@staticmethod
	def _make_convolution_layer(scope, input_tensor, kernel_size, output_channel, stride, padding, with_relu):
		with tf.variable_scope(scope):
			input_chanel = input_tensor.shape.as_list()[-1]
			kernel = tf.get_variable("kernel", shape=[kernel_size, kernel_size, input_chanel, output_channel],
				initializer=tf.contrib.layers.xavier_initializer(uniform=True),
			)
			bias = tf.get_variable("bias", shape=[output_channel],
				initializer=tf.contrib.layers.xavier_initializer(uniform=True),
			)

			convolution = tf.nn.conv2d(input_tensor, filter=kernel, strides=[1, stride, stride, 1], padding=padding)
			output = tf.nn.bias_add(convolution, bias)
			if with_relu: output = tf.nn.relu(output)
		return output

	@staticmethod
	def _make_fully_connected_layer(scope,  input_tensor, output_channel, with_relu):
		with tf.variable_scope(scope):
			input_chanel = input_tensor.shape.as_list()[-1]
			weight = tf.get_variable("weight", shape=[input_chanel, output_channel],
				initializer=tf.contrib.layers.xavier_initializer(uniform=True),
			)
			bias = tf.get_variable("bias", shape=[output_channel],
				initializer=tf.contrib.layers.xavier_initializer(uniform=True),
			)

			linear_combination = tf.matmul(input_tensor, weight)
			output = tf.nn.bias_add(linear_combination, bias)
			if with_relu: output = tf.nn.relu(output)
		return output

	def _setupSaver(self, checkpoint_path):
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)
			print "[Info] Create checkpoint path '%s'" % checkpoint_path
		else:
			print >> sys.stderr, "[Error] Checkpoint path '%s' already exists" % checkpoint_path
			exit(1)

		self._check_point_prefix = os.path.join(checkpoint_path,'model')
		self._saver = tf.train.Saver()

	def copy_variable_to(self, other):
		copy_operations = []
		for other_variable, self_variable in zip(other.trainable_variables, self.trainable_variables):
			copy_operations.append(tf.assign(other_variable, self_variable))
		self.session.run(copy_operations)

	def get_weight_decay_loss(self):
		return  self._weight_decay * tf.add_n([tf.nn.l2_loss(i) for i in self.trainable_variables])

	def loadModel(self, model_path):
		self.saver.restore(self.session, model_path)

	def saveModel(self, global_step):
		self.saver.saver(self.session, self.check_point_prefix, global_step=global_step)

if __name__ == "__main__":
	x = BasicModel(None, "test")
