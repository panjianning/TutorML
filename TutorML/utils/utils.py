import numpy as np
import numbers

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot to used to seed a numpy.random \
        RandomState instance" % seed)