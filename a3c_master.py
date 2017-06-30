import tensorflow as tf
import threading
from actor_critic_model import ActorCriticModel
from a3c_worker import A3cWorker
from environment import Environment

class A3cMaster():
	def __init__(self, gpu_id, environment, worker_count, max_elapsed_time):
		self._lock = threading.Lock()
		self._elasped_time = 0
		self._max_elapsed_time = max_elapsed_time
		with tf.device('/gpu:%d' % gpu_id):
			config=tf.ConfigProto()
			config.allow_soft_placement=True
			config.gpu_options.allow_growth=True
			self._session = tf.Session(config=config)
			self._optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
			self._global_model =  ActorCriticModel(
				session=self._session,
				scope="global",
				checkpoint_path="log",
				weight_decay=1e-3,
				policy_regular_factor=1e-3,
			)

		self._worker_list = []
		for i in range(worker_count):
			self._worker_list.append(A3cWorker(self, environment, i))

	def copy_model_weight(self, other_model):
		with self._lock:
			self._global_model.copy_variable_to(other_model)

	@property
	def session(self):
		return self._session

	def start_training(self):
		self.session.run(tf.global_variables_initializer())
		for thread in self._worker_list:
			thread.start()

		for thread in self._worker_list:
			thread.join()

	def update_global_model(self, elapsed_time, gradients):
		with self._lock:
			self._elasped_time += elapsed_time
			self._session.run(self._optimizer.apply_gradients(zip(gradients, self._global_model.trainable_variables)))
		return self._elasped_time <= self._max_elapsed_time