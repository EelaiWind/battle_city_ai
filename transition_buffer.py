from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple("Transition", ['state', 'action', 'reward', 'next_state', 'terminate'])

class TransitionBuffer():
	def __init__(self, max_size):
		self._max_size = max_size
		self._buffer = deque(maxlen = max_size)

	def clear(self):
		self._buffer.clear()

	@property
	def count(self):
		return len(self._buffer)

	@property
	def max_size(self):
		return self._max_size

	def pop(self):
		return self._buffer.pop()

	def push(self, state, action, reward, next_state, terminate):
		self._buffer.append(Transition(state, action, reward, next_state, terminate))

	def sample(self, batch_size):
		sample_size = min(batch_size, len(self._buffer))
		batch = random.sample(self._buffer, sample_size)
		return map(lambda i: np.stack(i, axis=0), zip(*batch))

if __name__ == "__main__":
	np.set_printoptions(linewidth=np.inf)
	experience_replay = TransitionBuffer(15)
	for i in range(20):
		experience_replay.push("s_%d" % i, "a_%d" % i, "r_%d" % i, "s_%d" % (i+1), bool(random.getrandbits(1)))

	state, action, reward, next_state, terminate = experience_replay.sample(15)
	print state
	print action
	print reward
	print next_state
	print terminate


