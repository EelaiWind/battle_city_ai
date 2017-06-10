import tensorflow as tf
from base_model import BaseModel
import numpy as np

class ActorCriticModel(BaseModel):
	MAP_SIZE = 28
	MAP_FEATURE = 19
	ACTION_DIMENSION = 8
	def __init__(self, session, scope, checkpoint_path, weight_decay, policy_regular_factor):
		self._policy_regular_factor = policy_regular_factor
		super(ActorCriticModel, self).__init__(session, scope, checkpoint_path, weight_decay=weight_decay)

	def _buildModel(self):
		if hasattr(self, '_policy'): return
		with tf.variable_scope(self.scope):
			self._spatial_input = tf.placeholder(tf.float32, shape=[None, ActorCriticModel.MAP_SIZE, ActorCriticModel.MAP_SIZE, ActorCriticModel.MAP_FEATURE])

	 		conv_1 = self._make_convolution_layer('conv_1', self._spatial_input, kernel_size=5, output_channel=64, stride=2, padding='SAME', with_relu=True)
	 		print conv_1.name, conv_1.shape
	 		conv_2 = self._make_convolution_layer('conv_2', conv_1, kernel_size=3, output_channel=64, stride=1, padding='SAME', with_relu=True)
	 		print conv_2.name, conv_2.shape
	 		conv_3 = self._make_convolution_layer('conv_3', conv_2, kernel_size=3, output_channel=64, stride=1, padding='SAME', with_relu=True)
	 		print conv_3.name, conv_3.shape
	 		conv_4 = self._make_convolution_layer('conv_4', conv_3, kernel_size=3, output_channel=64, stride=2, padding='SAME', with_relu=True)
	 		print conv_4.name, conv_4.shape
	 		flatten = tf.contrib.layers.flatten(conv_4)
	 		print flatten.name, flatten.shape
			fc_4 = self._make_fully_connected_layer('fc_4', flatten, output_channel=256, with_relu=True)
	 		print fc_4.name, fc_4.shape
			fc_5 = self._make_fully_connected_layer('fc_5', fc_4, output_channel=ActorCriticModel.ACTION_DIMENSION, with_relu=False)
			self._policy = tf.nn.softmax(fc_5)
	 		print self._policy.name, self._policy.shape
			fc_6 = self._make_fully_connected_layer('fc_6', flatten, output_channel=256, with_relu=True)
	 		print fc_6.name, fc_6.shape
			self._state_value = self._make_fully_connected_layer('fc_7', fc_6, output_channel=1, with_relu=False)
	 		print self._state_value.name, self._state_value.shape

	def _build_optimizer(self):
			log_policy = tf.log(tf.clip_by_value(self._policy, 1e-20, 1.0))
			
			self._selected_action = tf.placeholder(tf.int32, shape=[None, 1])
			selected_action_log_probability = tf.reduce_sum(log_policy * tf.one_hot(self._selected_action, ActorCriticModel.ACTION_DIMENSION), axis=1)

			self._sampled_return = tf.placeholder(tf.float32, shape=[None, 1])
			advantage = tf.subtract(self._sampled_return, tf.stop_gradient(self._state_value))

			policy_score = tf.reduce_sum(selected_action_log_probability * advantage, axis=1 )							# Maximize
			entropy_cost = -tf.reduce_sum(self._policy * log_policy, axis=1) 											# Maximize
			value_loss = 0.5 * tf.reduce_sum(tf.squared_difference(self._sampled_return, self._state_value), axis=1)	# minimize
			total_loss = tf.reduce_mean(tf.add_n([-policy_score, -self._policy_regular_factor * entropy_cost, value_loss]), axis=0) + self.get_weight_decay_loss()

			self._update_operation = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(total_loss)

	def forward(self, spatial_input):
		return self.session.run([self._policy, self._state_value], feed_dict={
			self._spatial_input: spatial_input,
		})

	def update(self, learning_rate, spatial_input, actions, sampled_return):
		self.session.run(self._update_operation, feed_dict={
			self._spatial_input: spatial_input,
			self._selected_action: actions,
			self._sampled_return: sampled_return,
			self._learning_rate: learning_rate,
		})

if __name__ == "__main__":
	sess = tf.Session()
	model = ActorCriticModel(sess, scope="test", checkpoint_path="tmp", weight_decay=1e-3, policy_regular_factor=1e-3)
	for i in model.trainable_variables:
		print i.name, i.shape