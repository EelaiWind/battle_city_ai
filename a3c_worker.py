import tensorflow as tf
import threading
from actor_critic_model import ActorCriticModel
from transition_buffer import TransitionBuffer
from collections import deque
import numpy as np

class A3cWorker(threading.Thread):
	UPDATE_INTERVAL = 64
	GAMMA = 0.9
	def __init__(self, master, environment, thread_id, device_id=None):
		super(A3cWorker, self).__init__(name="worker_%d" % thread_id)
		self._environment = environment.copy()
		self._master = master
		device_id = device_id or thread_id
		with tf.device('/cpu:%d' % device_id):
			self._local_model =  ActorCriticModel(
				session=master.session,
				scope=self.name,
				weight_decay=1e-3,
				policy_regular_factor=1e-3,
			)

	def run(self):
		transition_buffer = TransitionBuffer(A3cWorker.UPDATE_INTERVAL)
		batch_states = []
		batch_actions = []
		batch_returns = []

		state = self._environment.reset()
		while True:
			elapsed_time = 0
			transition_buffer.clear()
			self._master.copy_model_weight(self._local_model)
			while elapsed_time < A3cWorker.UPDATE_INTERVAL:
				policy, _ = self._local_model.forward(np.expand_dims(state, axis=0))
				action = np.argmax(np.random.multinomial(1, policy[0], size=1)[0])
				next_state, reward, terminate, _ = self._environment.step(action)
				transition_buffer.push(state, action, reward, next_state, terminate)
				elapsed_time += 1
				if terminate:
					state = self._environment.reset()
					break
				else:
					state = next_state

			if terminate:
				decay_return = 0
			else:
				_, decay_return = self._local_model.forward(np.expand_dims(state, axis=0))

			del batch_states [:]
			del batch_actions [:]
			del batch_returns [:]
			while transition_buffer.count > 0:
				transition = transition_buffer.pop()
				batch_states.append(transition.state)
				batch_actions.append(transition.action)
				batch_returns.append(decay_return)
				decay_return = transition.reward + A3cWorker.GAMMA*decay_return

			gradients = self._local_model.calculate_gradients(batch_states, np.array(batch_actions).reshape([-1,1]), np.array(batch_returns).reshape([-1,1]))
			if self._master.update_global_model(elapsed_time, gradients) is False: break