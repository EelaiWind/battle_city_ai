import tensorflow as tf
from basic_model import BasicModel
import numpy as np

class ActorCriticModel(BasicModel):
	MAP_WIDTH = 160
	MAP_HEIGHT = 210
	MAP_FEATURE = 3
	ACTION_DIMENSION = 4
	def __init__(self, session, scope, checkpoint_path=None, weight_decay=0.0, policy_regular_factor=0.0):
		self._policy_regular_factor = policy_regular_factor
		super(ActorCriticModel, self).__init__(session, scope, checkpoint_path, weight_decay)

	def _build_loss(self):
		if hasattr(self, '_gradients'): return
		with tf.variable_scope(self.scope):
			log_policy = tf.log(tf.clip_by_value(self._policy, 1e-20, 1.0))

			self._selected_action = tf.placeholder(tf.int32, shape=[None, 1])
			selected_action_log_probability = tf.reduce_sum(log_policy * tf.one_hot(self._selected_action, ActorCriticModel.ACTION_DIMENSION), axis=1)

			self._sampled_return = tf.placeholder(tf.float32, shape=[None, 1])
			advantage = tf.subtract(self._sampled_return, tf.stop_gradient(self._state_value))

			policy_score = tf.reduce_sum(selected_action_log_probability * advantage, axis=1 )							# Maximize
			entropy_cost = -tf.reduce_sum(self._policy * log_policy, axis=1) 											# Maximize
			value_loss = 0.5 * tf.reduce_sum(tf.squared_difference(self._sampled_return, self._state_value), axis=1)	# minimize
			total_loss = tf.reduce_mean(tf.add_n([-policy_score, -self._policy_regular_factor * entropy_cost, value_loss]), axis=0) + self.get_weight_decay_loss()

			self._gradients = tf.gradients(total_loss, self.trainable_variables)

	def _build_model(self):
		if hasattr(self, '_policy'): return
		with tf.variable_scope(self.scope):
			self._spatial_input = tf.placeholder(tf.float32, shape=[None, ActorCriticModel.MAP_HEIGHT, ActorCriticModel.MAP_WIDTH, ActorCriticModel.MAP_FEATURE])

	 		conv_1 = BasicModel._make_convolution_layer('conv_1', self._spatial_input, kernel_size=5, output_channel=64, stride=1, padding='SAME', with_relu=True)
	 		conv_2 = BasicModel._make_convolution_layer('conv_2', conv_1, kernel_size=3, output_channel=64, stride=2, padding='SAME', with_relu=True)
	 		conv_3 = BasicModel._make_convolution_layer('conv_3', conv_2, kernel_size=3, output_channel=128, stride=2, padding='SAME', with_relu=True)
	 		conv_4 = BasicModel._make_convolution_layer('conv_4', conv_3, kernel_size=3, output_channel=256, stride=2, padding='SAME', with_relu=True)
	 		flatten = tf.contrib.layers.flatten(conv_4)
			fc_4 = BasicModel._make_fully_connected_layer('fc_4', flatten, output_channel=1024, with_relu=True)
			fc_5 = BasicModel._make_fully_connected_layer('fc_5', fc_4, output_channel=ActorCriticModel.ACTION_DIMENSION, with_relu=False)
			self._policy = tf.nn.softmax(fc_5)
			fc_6 = BasicModel._make_fully_connected_layer('fc_6', flatten, output_channel=1024, with_relu=True)
			self._state_value = BasicModel._make_fully_connected_layer('fc_7', fc_6, output_channel=1, with_relu=False)

	def calculate_gradients(self, spatial_input, actions, sampled_return):
		return self.session.run(self._gradients, feed_dict={
			self._spatial_input: spatial_input,
			self._selected_action: actions,
			self._sampled_return: sampled_return,
		})

	def forward(self, spatial_input):
		return self.session.run([self._policy, self._state_value], feed_dict={
			self._spatial_input: spatial_input,
		})

if __name__ == "__main__":
	sess = tf.Session()
	model = ActorCriticModel(sess, scope="test", checkpoint_path="tmp", weight_decay=1e-3, policy_regular_factor=1e-3)
	for i in model.trainable_variables:
		print i.name, i.shape